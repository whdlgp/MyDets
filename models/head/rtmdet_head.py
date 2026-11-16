# Adapted from MMDetection and MMCV (Apache-2.0). See THIRD_PARTY_NOTICES.txt.
from collections.abc import Callable
import math

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.autograd import Function
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation, batched_nms, box_convert

from models.base import Detections, Head, Targets


def reduce_mean(tensor: Tensor) -> Tensor:
    """Average a tensor over the processes of distributed training, unchanged otherwise."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class SigmoidGeometricMean(Function):
    """Geometric mean of two sigmoids with the analytical gradient, which avoids NaN for very small inputs."""

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        """Compute sqrt(sigmoid(x) * sigmoid(y))."""
        x_sigmoid = x.sigmoid()
        y_sigmoid = y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the gradients of both inputs."""
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        return grad_output * z * (1 - x_sigmoid) / 2, grad_output * z * (1 - y_sigmoid) / 2


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise conv block followed by a pointwise conv block, each with batch norm and activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation_layer: Callable[..., nn.Module]) -> None:
        """Initialize the block.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            kernel_size: depthwise conv kernel size.
            activation_layer: activation layer.
        """
        super().__init__(
            Conv2dNormActivation(in_channels, in_channels, kernel_size, groups=in_channels, activation_layer=activation_layer),
            Conv2dNormActivation(in_channels, out_channels, 1, activation_layer=activation_layer)
        )


class RTMDetSepBNHead(Head):
    """RTMDet head with convs shared over levels and batch norms separated per level, with dynamic soft label assignment."""

    def __init__(self, in_channels: list[int], in_strides: list[int], num_classes: int, hidden_channels: int = 256, num_convs: int = 2, share_conv: bool = True, depthwise: bool = False, exp_on_reg: bool = True, with_objectness: bool = False, pred_kernel_size: int = 1, activation: Callable[..., nn.Module] = nn.SiLU, topk: int = 13, soft_center_radius: float = 3.0, iou_weight: float = 3.0, score_threshold: float = 0.001, pre_nms_topk: int = 30000, nms_threshold: float = 0.65, max_detections: int = 300) -> None:
        """Initialize the head.

        Args:
            in_channels: channels of each input feature map, all equal.
            in_strides: stride of each input feature map.
            num_classes: number of object classes.
            hidden_channels: channels of the branch convs.
            num_convs: number of convs in each branch.
            share_conv: whether the branch convs are shared over levels, with batch norms still per level.
            depthwise: whether the branch convs are depthwise separable, not supported with share_conv.
            exp_on_reg: whether the box distances are the exponential of the predictions.
            with_objectness: whether an objectness prediction joins the class scores by geometric mean.
            pred_kernel_size: kernel size of the prediction convs.
            activation: activation layer.
            topk: number of top IoUs summed for the dynamic k of each gt.
            soft_center_radius: radius of the soft center prior in strides.
            iou_weight: weight of the IoU cost.
            score_threshold: minimum score to keep a box.
            pre_nms_topk: maximum boxes kept per level before NMS.
            nms_threshold: IoU threshold of NMS.
            max_detections: maximum detections per image.
        """
        super().__init__(in_channels, in_strides, num_classes)

        if len(set(in_channels)) != 1:
            raise ValueError(f"the head expects equal channels over levels, got {in_channels}")

        if share_conv and depthwise:
            raise ValueError("share_conv does not support depthwise convs")

        self.exp_on_reg = exp_on_reg
        self.with_objectness = with_objectness
        self.topk = topk
        self.soft_center_radius = soft_center_radius
        self.iou_weight = iou_weight
        self.score_threshold = score_threshold
        self.pre_nms_topk = pre_nms_topk
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        conv = DepthwiseSeparableConv if depthwise else Conv2dNormActivation
        padding = pred_kernel_size // 2
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_obj = nn.ModuleList()

        for _ in in_strides:
            channels = [in_channels[0]] + [hidden_channels] * num_convs
            self.cls_convs.append(nn.Sequential(*[conv(c, hidden_channels, 3, activation_layer=activation) for c in channels[:-1]]))
            self.reg_convs.append(nn.Sequential(*[conv(c, hidden_channels, 3, activation_layer=activation) for c in channels[:-1]]))
            self.rtm_cls.append(nn.Conv2d(hidden_channels, num_classes, pred_kernel_size, padding=padding))
            self.rtm_reg.append(nn.Conv2d(hidden_channels, 4, pred_kernel_size, padding=padding))

            if with_objectness:
                self.rtm_obj.append(nn.Conv2d(hidden_channels, 1, pred_kernel_size, padding=padding))

        if share_conv:
            self.share_convs()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # the class and objectness biases start from a prior probability of 0.01
        bias = -math.log((1 - 0.01) / 0.01)

        for pred in [*self.rtm_cls, *self.rtm_obj]:
            nn.init.constant_(pred.bias, bias)

    def share_convs(self) -> None:
        """Make every level use the convs of the first level, keeping its own batch norms."""
        for branch in [self.cls_convs, self.reg_convs]:
            for level_convs in branch[1:]:
                for block, first in zip(level_convs, branch[0]):
                    block[0] = first[0]

    def forward(self, features: list[Tensor], targets: list[Targets] | None = None) -> dict[str, Tensor]:
        """Predict the raw outputs of every location.

        Args:
            features: neck feature maps, from high to low resolution.
            targets: unused.

        Returns:
            Raw class logits (B, N, C), ltrb distances in pixels (B, N, 4), priors (N, 3) as x, y and stride
            in pixels, level of each prior (N,), and the input size (width, height).
        """
        cls_logits, distances, priors, levels = [], [], [], []

        for level, (x, stride) in enumerate(zip(features, self.in_strides)):
            cls_feat = self.cls_convs[level](x)
            reg_feat = self.reg_convs[level](x)
            cls_score = self.rtm_cls[level](cls_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[level](reg_feat)
                cls_score = self.inverse_sigmoid(SigmoidGeometricMean.apply(cls_score, objectness))

            reg = self.rtm_reg[level](reg_feat)
            reg = reg.exp() if self.exp_on_reg else reg
            cls_logits.append(cls_score.flatten(2))
            distances.append((reg * stride).flatten(2))

            height, width = x.shape[-2:]
            shift_y, shift_x = torch.meshgrid(torch.arange(height, device=x.device) * stride, torch.arange(width, device=x.device) * stride, indexing="ij")
            strides = torch.full_like(shift_x, stride)
            priors.append(torch.stack([shift_x, shift_y, strides], -1).reshape(-1, 3).to(x.dtype))
            levels.append(torch.full((height * width,), level, device=x.device))

        # exact only for inputs that are multiples of the strides
        image_size = features[0].new_tensor([features[0].shape[-1] * self.in_strides[0], features[0].shape[-2] * self.in_strides[0]])

        return {
            "cls_logits": torch.cat(cls_logits, 2).transpose(1, 2),
            "distances": torch.cat(distances, 2).transpose(1, 2),
            "priors": torch.cat(priors),
            "levels": torch.cat(levels),
            "image_size": image_size
        }

    @staticmethod
    def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
        """Inverse of the sigmoid with the input clamped away from 0 and 1."""
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    @staticmethod
    def distance_to_boxes(points: Tensor, distances: Tensor) -> Tensor:
        """Convert ltrb distances from the points to xyxy boxes."""
        return torch.cat([points - distances[..., :2], points + distances[..., 2:]], -1)

    def decode(self, outputs: dict[str, Tensor]) -> list[Detections]:
        """Keep the top scoring prior-class pairs of each level above the threshold, clip, and apply class-wise NMS.

        Args:
            outputs: raw outputs of forward.

        Returns:
            Detections of each image.
        """
        priors, levels = outputs["priors"], outputs["levels"]
        width, height = outputs["image_size"].tolist()
        scale = priors.new_tensor([width, height, width, height])
        detections = []

        for logits, distances in zip(outputs["cls_logits"], outputs["distances"]):
            kept_distances, kept_points, scores, labels = [], [], [], []

            for level in range(len(self.in_strides)):
                in_level = levels == level
                level_scores = logits[in_level].sigmoid()
                candidates = torch.nonzero(level_scores > self.score_threshold)
                level_scores, order = level_scores[candidates[:, 0], candidates[:, 1]].sort(descending=True)
                indices, classes = candidates[order[:self.pre_nms_topk]].unbind(1)
                kept_distances.append(distances[in_level][indices])
                kept_points.append(priors[in_level][indices, :2])
                scores.append(level_scores[:self.pre_nms_topk])
                labels.append(classes)

            boxes = self.distance_to_boxes(torch.cat(kept_points), torch.cat(kept_distances))
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
            scores, labels = torch.cat(scores), torch.cat(labels)

            # boxes without area are removed before NMS
            valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
            boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            keep = batched_nms(boxes, scores, labels, self.nms_threshold)[:self.max_detections]
            detections.append(Detections(box_convert(boxes[keep], "xyxy", "cxcywh") / scale, scores[keep], labels[keep]))

        return detections

    def loss(self, outputs: dict[str, Tensor], targets: list[Targets], state: dict[str, int | float]) -> dict[str, Tensor]:
        """Assign the priors with dynamic soft labels and compute the quality focal loss and the IoU weighted GIoU loss.

        Args:
            outputs: raw outputs of forward.
            targets: ground truth of each image.
            state: unused.

        Returns:
            Losses, with the total under the key "loss".
        """
        cls_logits, distances, priors = outputs["cls_logits"], outputs["distances"], outputs["priors"]
        width, height = outputs["image_size"].tolist()
        scale = priors.new_tensor([width, height, width, height])
        boxes = self.distance_to_boxes(priors[:, :2], distances)
        labels, gt_boxes, assign_metrics = [], [], []

        for i, target in enumerate(targets):
            image_labels, image_boxes, image_metrics = self.assign(box_convert(target.boxes, "cxcywh", "xyxy") * scale, target.labels, boxes[i].detach(), cls_logits[i].detach(), priors)
            labels.append(image_labels)
            gt_boxes.append(image_boxes)
            assign_metrics.append(image_metrics)

        labels = torch.cat(labels)
        gt_boxes = torch.cat(gt_boxes)
        assign_metrics = torch.cat(assign_metrics)
        pos = labels < self.num_classes

        # the normalizer is the sum of the IoU targets, averaged over the processes of distributed training
        avg_factor = reduce_mean(assign_metrics.sum()).clamp_(min=1).item()

        # each loss sum is first divided by 1 plus the float32 eps, then by the sum of the IoU targets
        eps = torch.finfo(torch.float32).eps
        loss_cls = self.quality_focal_loss(cls_logits.reshape(-1, self.num_classes), labels, assign_metrics).sum() / (1.0 + eps) / avg_factor
        loss_box = boxes.sum() * 0

        if pos.any():
            # the GIoU loss weight is 2
            losses = self.giou_loss(boxes.reshape(-1, 4)[pos], gt_boxes[pos]) * assign_metrics[pos]
            loss_box = 2.0 * losses.sum() / (1.0 + eps) / avg_factor

        return {"loss": loss_cls + loss_box, "loss_cls": loss_cls, "loss_box": loss_box}

    def quality_focal_loss(self, logits: Tensor, labels: Tensor, scores: Tensor) -> Tensor:
        """Quality focal loss with beta 2, where the positive target is the IoU.

        Args:
            logits: class logits of shape (N, C).
            labels: class of each prior, num_classes for background, of shape (N,).
            scores: IoU target of each prior, of shape (N,).

        Returns:
            Loss of each prior, of shape (N,).
        """
        probs = logits.sigmoid()
        loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none") * probs.pow(2.0)
        pos = torch.nonzero(labels < self.num_classes)[:, 0]
        pos_labels = labels[pos]
        factor = scores[pos] - probs[pos, pos_labels]
        loss[pos, pos_labels] = F.binary_cross_entropy_with_logits(logits[pos, pos_labels], scores[pos], reduction="none") * factor.abs().pow(2.0)
        return loss.sum(1)

    @staticmethod
    def giou_loss(boxes: Tensor, gt_boxes: Tensor, eps: float = 1e-6) -> Tensor:
        """One minus the GIoU of aligned xyxy boxes, with the union and the enclosing area clamped at eps.

        Args:
            boxes: predicted boxes of shape (M, 4).
            gt_boxes: target boxes of shape (M, 4).
            eps: minimum of the union and the enclosing area.

        Returns:
            Loss of each box, of shape (M,).
        """
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        overlap = (torch.min(boxes[:, 2:], gt_boxes[:, 2:]) - torch.max(boxes[:, :2], gt_boxes[:, :2])).clamp(min=0).prod(1)
        union = torch.clamp(area + gt_area - overlap, min=eps)
        enclosing = (torch.max(boxes[:, 2:], gt_boxes[:, 2:]) - torch.min(boxes[:, :2], gt_boxes[:, :2])).clamp(min=0).prod(1)
        enclosing = torch.clamp(enclosing, min=eps)
        ious = overlap / union
        return 1 - (ious - (enclosing - union) / enclosing)

    @torch.no_grad()
    def assign(self, gt_boxes: Tensor, gt_labels: Tensor, boxes: Tensor, cls_logits: Tensor, priors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Assign the priors of one image with dynamic soft labels.

        Args:
            gt_boxes: xyxy boxes in pixels of shape (G, 4).
            gt_labels: classes of shape (G,).
            boxes: predicted xyxy boxes of shape (N, 4).
            cls_logits: class logits of shape (N, C).
            priors: x, y and stride in pixels of shape (N, 3).

        Returns:
            Label of each prior (num_classes for background), matched xyxy gt box, and IoU of the positives (zero elsewhere).
        """
        num_priors, num_gt = len(priors), len(gt_boxes)
        labels = gt_labels.new_full((num_priors,), self.num_classes, dtype=torch.long)
        matched_boxes = torch.zeros_like(boxes)
        metrics = boxes.new_zeros(num_priors)

        if num_gt == 0:
            return labels, matched_boxes, metrics

        # candidates are the priors inside a gt
        points = priors[:, :2]
        deltas = torch.cat([points[:, None] - gt_boxes[:, :2], gt_boxes[:, 2:] - points[:, None]], -1)
        valid = (deltas.min(-1).values > 0).any(1)

        if not valid.any():
            return labels, matched_boxes, metrics

        valid_boxes = boxes[valid]
        valid_logits = cls_logits[valid]
        valid_priors = priors[valid]
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
        distance = (valid_priors[:, None, :2] - gt_centers[None]).pow(2).sum(-1).sqrt() / valid_priors[:, 2:]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        ious = self.pairwise_ious(valid_boxes, gt_boxes)
        iou_cost = -torch.log(ious + 1e-7) * self.iou_weight
        num_valid = len(valid_boxes)
        gt_onehot = F.one_hot(gt_labels.long(), self.num_classes).float()[None].repeat(num_valid, 1, 1)
        valid_logits = valid_logits[:, None].repeat(1, num_gt, 1)
        soft_label = gt_onehot * ious[..., None]
        factor = soft_label - valid_logits.sigmoid()
        soft_cls_cost = (F.binary_cross_entropy_with_logits(valid_logits, soft_label, reduction="none") * factor.abs().pow(2.0)).sum(-1)
        cost = soft_cls_cost + iou_cost + soft_center_prior

        # dynamic k from the sum of the top IoUs of each gt
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        topk_ious, _ = torch.topk(ious, min(self.topk, num_valid), 0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_index in range(num_gt):
            _, positions = torch.topk(cost[:, gt_index], dynamic_ks[gt_index].item(), largest=False)
            matching[positions, gt_index] = 1

        # a prior matched to several gts keeps the one with the lowest cost
        multiple = matching.sum(1) > 1

        if multiple.any():
            lowest = cost[multiple].argmin(1)
            matching[multiple] = 0
            matching[multiple, lowest] = 1

        matched = matching.sum(1) > 0
        valid[valid.clone()] = matched
        matched_gt = matching[matched].argmax(1)
        labels[valid] = gt_labels[matched_gt].long()
        matched_boxes[valid] = gt_boxes[matched_gt]
        metrics[valid] = (matching * ious).sum(1)[matched]
        return labels, matched_boxes, metrics

    @staticmethod
    def pairwise_ious(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-6) -> Tensor:
        """IoU between every pair of xyxy boxes, with the union clamped at eps.

        Args:
            boxes1: boxes of shape (N, 4).
            boxes2: boxes of shape (G, 4).
            eps: minimum of the union.

        Returns:
            IoU of shape (N, G).
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        overlap = (torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:]) - torch.max(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0).prod(-1)
        union = torch.clamp(area1[:, None] + area2[None] - overlap, min=eps)
        return overlap / union
