# Adapted from YOLOX (Apache-2.0). See THIRD_PARTY_NOTICES.txt.
from collections.abc import Callable
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import batched_nms, box_convert, nms

from models.base import Detections, Head, Targets


class ConvBlock(nn.Sequential):
    """Conv without bias, batch norm with eps 1e-3 and momentum 0.03, and activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: Callable[..., nn.Module], groups: int = 1) -> None:
        """Initialize the block.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            kernel_size: conv kernel size, padded to keep the size.
            activation: activation layer.
            groups: conv groups.
        """
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
            activation(inplace=True)
        )


class DepthwiseConvBlock(nn.Sequential):
    """Depthwise conv block followed by a pointwise conv block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: Callable[..., nn.Module]) -> None:
        """Initialize the block.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            kernel_size: depthwise conv kernel size.
            activation: activation layer.
        """
        super().__init__(
            ConvBlock(in_channels, in_channels, kernel_size, activation, groups=in_channels),
            ConvBlock(in_channels, out_channels, 1, activation)
        )


class YOLOXHead(Head):
    """Decoupled anchor-free head of YOLOX with SimOTA assignment."""

    def __init__(self, in_channels: list[int], in_strides: list[int], num_classes: int, hidden_channels: int = 256, depthwise: bool = False, activation: Callable[..., nn.Module] = nn.SiLU, l1_epochs: int = 15, score_threshold: float = 0.01, nms_threshold: float = 0.65, class_agnostic: bool = False) -> None:
        """Initialize the head.

        Args:
            in_channels: channels of each input feature map.
            in_strides: stride of each input feature map.
            num_classes: number of object classes.
            hidden_channels: channels of the stem and branch convs.
            depthwise: whether the branch convs are depthwise separable.
            activation: activation layer.
            l1_epochs: the L1 loss is added from the epoch where epoch + 1 equals num_epochs - l1_epochs, as the original trainer does.
            score_threshold: minimum of objectness times class score to keep a box.
            nms_threshold: IoU threshold of NMS.
            class_agnostic: whether NMS ignores the classes.
        """
        super().__init__(in_channels, in_strides, num_classes)
        self.l1_epochs = l1_epochs
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.class_agnostic = class_agnostic
        conv = DepthwiseConvBlock if depthwise else ConvBlock
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for channels in in_channels:
            self.stems.append(ConvBlock(channels, hidden_channels, 1, activation))
            self.cls_convs.append(nn.Sequential(conv(hidden_channels, hidden_channels, 3, activation), conv(hidden_channels, hidden_channels, 3, activation)))
            self.reg_convs.append(nn.Sequential(conv(hidden_channels, hidden_channels, 3, activation), conv(hidden_channels, hidden_channels, 3, activation)))
            self.cls_preds.append(nn.Conv2d(hidden_channels, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hidden_channels, 4, 1))
            self.obj_preds.append(nn.Conv2d(hidden_channels, 1, 1))

        # the class and objectness biases start from a prior probability of 0.01
        bias = -math.log((1 - 0.01) / 0.01)

        for pred in [*self.cls_preds, *self.obj_preds]:
            nn.init.constant_(pred.bias, bias)

    def forward(self, features: list[Tensor], targets: list[Targets] | None = None) -> dict[str, Tensor]:
        """Predict the raw outputs of every location.

        Args:
            features: neck feature maps, from high to low resolution.
            targets: unused.

        Returns:
            Raw class logits (B, N, C), objectness logits (B, N), box deltas (B, N, 4),
            priors (N, 3) as grid x, grid y and stride, and the input size (width, height).
        """
        cls_logits, obj_logits, box_deltas, priors = [], [], [], []

        for i, (x, stride) in enumerate(zip(features, self.in_strides)):
            x = self.stems[i](x)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            cls_logits.append(self.cls_preds[i](cls_feat).flatten(2))
            obj_logits.append(self.obj_preds[i](reg_feat).flatten(2))
            box_deltas.append(self.reg_preds[i](reg_feat).flatten(2))
            height, width = x.shape[-2:]
            grid_y, grid_x = torch.meshgrid(torch.arange(height, device=x.device), torch.arange(width, device=x.device), indexing="ij")
            strides = torch.full_like(grid_x, stride)
            priors.append(torch.stack([grid_x, grid_y, strides], -1).reshape(-1, 3).to(x.dtype))

        # exact only for inputs that are multiples of the strides
        image_size = features[0].new_tensor([features[0].shape[-1] * self.in_strides[0], features[0].shape[-2] * self.in_strides[0]])

        return {
            "cls_logits": torch.cat(cls_logits, 2).transpose(1, 2),
            "obj_logits": torch.cat(obj_logits, 2)[:, 0],
            "box_deltas": torch.cat(box_deltas, 2).transpose(1, 2),
            "priors": torch.cat(priors),
            "image_size": image_size
        }

    def decode(self, outputs: dict[str, Tensor]) -> list[Detections]:
        """Keep the best class of each location above the score threshold and apply NMS.

        Args:
            outputs: raw outputs of forward.

        Returns:
            Detections of each image.
        """
        boxes = self.boxes(outputs["box_deltas"], outputs["priors"])
        scores, labels = outputs["cls_logits"].sigmoid().max(-1)
        scores = scores * outputs["obj_logits"].sigmoid()
        width, height = outputs["image_size"].tolist()
        scale = boxes.new_tensor([width, height, width, height])
        detections = []

        for image_boxes, image_scores, image_labels in zip(boxes, scores, labels):
            keep = image_scores >= self.score_threshold
            image_boxes, image_scores, image_labels = image_boxes[keep], image_scores[keep], image_labels[keep]
            xyxy = box_convert(image_boxes, "cxcywh", "xyxy")

            if self.class_agnostic:
                keep = nms(xyxy, image_scores, self.nms_threshold)
            else:
                keep = batched_nms(xyxy, image_scores, image_labels, self.nms_threshold)

            detections.append(Detections(image_boxes[keep] / scale, image_scores[keep], image_labels[keep]))

        return detections

    def loss(self, outputs: dict[str, Tensor], targets: list[Targets], state: dict[str, int | float]) -> dict[str, Tensor]:
        """Assign the locations with SimOTA and compute the IoU, objectness, class and L1 losses.

        Args:
            outputs: raw outputs of forward.
            targets: ground truth of each image.
            state: training progress, with "epoch" (0-based) and "num_epochs".

        Returns:
            Losses, with the total under the key "loss".
        """
        cls_logits, obj_logits, box_deltas, priors = outputs["cls_logits"], outputs["obj_logits"], outputs["box_deltas"], outputs["priors"]
        boxes = self.boxes(box_deltas, priors)
        width, height = outputs["image_size"].tolist()
        scale = boxes.new_tensor([width, height, width, height])
        use_l1 = state["epoch"] + 1 >= state["num_epochs"] - self.l1_epochs
        cls_targets, reg_targets, l1_targets, obj_targets, fg_masks = [], [], [], [], []
        num_fg = 0

        for i, target in enumerate(targets):
            gt_boxes = target.boxes * scale
            fg_mask = boxes.new_zeros(len(priors), dtype=torch.bool)

            if len(gt_boxes) == 0:
                cls_targets.append(boxes.new_zeros(0, self.num_classes))
                reg_targets.append(boxes.new_zeros(0, 4))
                l1_targets.append(boxes.new_zeros(0, 4))
                obj_targets.append(boxes.new_zeros(len(priors)))
                fg_masks.append(fg_mask)
                continue

            fg_mask, matched_gt, matched_ious = self.assign(gt_boxes, target.labels, boxes[i], cls_logits[i], obj_logits[i], priors)
            num_fg += len(matched_gt)
            cls_targets.append(F.one_hot(target.labels[matched_gt].long(), self.num_classes) * matched_ious[:, None])
            reg_targets.append(gt_boxes[matched_gt])
            l1_targets.append(self.l1_targets(gt_boxes[matched_gt], priors[fg_mask]))
            obj_targets.append(fg_mask.to(boxes.dtype))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets)
        reg_targets = torch.cat(reg_targets)
        obj_targets = torch.cat(obj_targets)
        fg_masks = torch.cat(fg_masks)
        num_fg = max(num_fg, 1)

        loss_iou = self.iou_loss(boxes.reshape(-1, 4)[fg_masks], reg_targets).sum() / num_fg
        loss_obj = F.binary_cross_entropy_with_logits(obj_logits.reshape(-1), obj_targets, reduction="sum") / num_fg
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits.reshape(-1, self.num_classes)[fg_masks], cls_targets.to(cls_logits.dtype), reduction="sum") / num_fg
        loss_l1 = boxes.new_zeros(())

        if use_l1:
            loss_l1 = F.l1_loss(box_deltas.reshape(-1, 4)[fg_masks], torch.cat(l1_targets), reduction="sum") / num_fg

        # the IoU loss weight is 5
        loss_iou = 5.0 * loss_iou
        return {"loss": loss_iou + loss_obj + loss_cls + loss_l1, "loss_iou": loss_iou, "loss_obj": loss_obj, "loss_cls": loss_cls, "loss_l1": loss_l1}

    def boxes(self, box_deltas: Tensor, priors: Tensor) -> Tensor:
        """Decode the deltas to cxcywh boxes in pixels.

        Args:
            box_deltas: raw box deltas of shape (B, N, 4).
            priors: grid x, grid y and stride of shape (N, 3).

        Returns:
            Boxes of shape (B, N, 4).
        """
        strides = priors[:, 2:]
        centers = (box_deltas[..., :2] + priors[:, :2]) * strides
        sizes = box_deltas[..., 2:].exp() * strides
        return torch.cat([centers, sizes], -1)

    def l1_targets(self, gt_boxes: Tensor, priors: Tensor) -> Tensor:
        """Encode the matched boxes as raw deltas for the L1 loss.

        Args:
            gt_boxes: matched cxcywh boxes in pixels of shape (M, 4).
            priors: grid x, grid y and stride of the matched locations, of shape (M, 3).

        Returns:
            Target deltas of shape (M, 4).
        """
        strides = priors[:, 2:]
        centers = gt_boxes[:, :2] / strides - priors[:, :2]
        sizes = torch.log(gt_boxes[:, 2:] / strides + 1e-8)
        return torch.cat([centers, sizes], -1)

    @torch.no_grad()
    def assign(self, gt_boxes: Tensor, gt_labels: Tensor, boxes: Tensor, cls_logits: Tensor, obj_logits: Tensor, priors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Assign the locations of one image with SimOTA.

        Args:
            gt_boxes: cxcywh boxes in pixels of shape (G, 4).
            gt_labels: classes of shape (G,).
            boxes: predicted cxcywh boxes in pixels of shape (N, 4).
            cls_logits: class logits of shape (N, C).
            obj_logits: objectness logits of shape (N,).
            priors: grid x, grid y and stride of shape (N, 3).

        Returns:
            Foreground mask (N,), matched gt index and IoU of each foreground location.
        """
        # candidates whose cell center is within 1.5 strides of a gt center
        strides = priors[:, 2]
        centers = (priors[:, :2] + 0.5) * strides[:, None]
        radius = 1.5 * strides
        offsets = (centers[None] - gt_boxes[:, None, :2]).abs()
        is_in_centers = (offsets < radius[None, :, None]).all(-1)
        fg_mask = is_in_centers.any(0)
        is_in_centers = is_in_centers[:, fg_mask]

        candidate_boxes = boxes[fg_mask]
        ious = self.pairwise_ious(gt_boxes, candidate_boxes)
        iou_cost = -torch.log(ious + 1e-8)
        probs = (cls_logits[fg_mask].float().sigmoid() * obj_logits[fg_mask, None].float().sigmoid()).sqrt()
        gt_onehot = F.one_hot(gt_labels.long(), self.num_classes).float()
        num_gt, num_candidates = ious.shape
        cls_cost = F.binary_cross_entropy(probs[None].expand(num_gt, -1, -1), gt_onehot[:, None].expand(-1, num_candidates, -1), reduction="none").sum(-1)
        cost = cls_cost + 3.0 * iou_cost + 1e6 * (~is_in_centers)

        # dynamic k from the sum of the top 10 IoUs of each gt
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        topk_ious, _ = torch.topk(ious, min(10, num_candidates), 1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_index in range(num_gt):
            _, positions = torch.topk(cost[gt_index], dynamic_ks[gt_index].item(), largest=False)
            matching[gt_index, positions] = 1

        # a location matched to several gts keeps the one with the lowest cost
        num_matches = matching.sum(0)

        if num_matches.max() > 1:
            multiple = num_matches > 1
            lowest = cost[:, multiple].argmin(0)
            matching[:, multiple] = 0
            matching[lowest, multiple] = 1

        matched = num_matches > 0
        fg_mask[fg_mask.clone()] = matched
        matched_gt = matching[:, matched].argmax(0)
        matched_ious = (matching * ious).sum(0)[matched]
        return fg_mask, matched_gt, matched_ious

    def pairwise_ious(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """IoU between every pair of cxcywh boxes, zero for boxes that only touch.

        Args:
            boxes1: boxes of shape (G, 4).
            boxes2: boxes of shape (N, 4).

        Returns:
            IoU of shape (G, N).
        """
        top_left = torch.max(boxes1[:, None, :2] - boxes1[:, None, 2:] / 2, boxes2[:, :2] - boxes2[:, 2:] / 2)
        bottom_right = torch.min(boxes1[:, None, :2] + boxes1[:, None, 2:] / 2, boxes2[:, :2] + boxes2[:, 2:] / 2)
        overlap = (top_left < bottom_right).to(top_left.dtype).prod(2)
        intersection = torch.prod(bottom_right - top_left, 2) * overlap
        area1 = torch.prod(boxes1[:, 2:], 1)
        area2 = torch.prod(boxes2[:, 2:], 1)
        return intersection / (area1[:, None] + area2 - intersection)

    def iou_loss(self, boxes: Tensor, gt_boxes: Tensor) -> Tensor:
        """One minus the squared IoU of aligned cxcywh boxes.

        Args:
            boxes: predicted boxes of shape (M, 4).
            gt_boxes: target boxes of shape (M, 4).

        Returns:
            Loss of each box, of shape (M,).
        """
        top_left = torch.max(boxes[:, :2] - boxes[:, 2:] / 2, gt_boxes[:, :2] - gt_boxes[:, 2:] / 2)
        bottom_right = torch.min(boxes[:, :2] + boxes[:, 2:] / 2, gt_boxes[:, :2] + gt_boxes[:, 2:] / 2)
        overlap = (top_left < bottom_right).to(top_left.dtype).prod(1)
        intersection = torch.prod(bottom_right - top_left, 1) * overlap
        union = torch.prod(boxes[:, 2:], 1) + torch.prod(gt_boxes[:, 2:], 1) - intersection
        iou = intersection / (union + 1e-16)
        return 1 - iou ** 2
