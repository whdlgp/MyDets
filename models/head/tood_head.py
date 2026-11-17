# Adapted from TOOD, MMDetection and MMCV (Apache-2.0). See THIRD_PARTY_NOTICES.txt.
from typing import Literal
import math

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn import functional as F
from torchvision.ops import batched_nms, box_convert, deform_conv2d

from models.base import Detections, Head, Targets


def reduce_mean(tensor: Tensor) -> Tensor:
    """Average a tensor over the processes of distributed training, unchanged otherwise."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class ModulatedDeformConv(nn.Module):
    """3x3 modulated deformable conv without bias, whose offsets and masks come from a zero initialized conv."""

    def __init__(self, in_channels: int, out_channels: int, deform_groups: int) -> None:
        """Initialize the conv.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            deform_groups: number of offset groups.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.conv_offset = nn.Conv2d(in_channels, deform_groups * 3 * 9, 3, padding=1)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the conv with the predicted offsets and sigmoid masks."""
        o1, o2, mask = torch.chunk(self.conv_offset(x), 3, dim=1)
        offset = torch.cat([o1, o2], 1)
        return deform_conv2d(x, offset, self.weight, padding=1, mask=mask.sigmoid())


class TaskDecomposition(nn.Module):
    """Layer attention over the stacked features followed by a 1x1 reduction conv, group norm and ReLU."""

    def __init__(self, hidden_channels: int, num_convs: int, down_rate: int) -> None:
        """Initialize the layer.

        Args:
            hidden_channels: channels of each stacked feature and of the output.
            num_convs: number of stacked features.
            down_rate: channel reduction of the layer attention.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_convs = num_convs
        in_channels = hidden_channels * num_convs
        self.la_conv1 = nn.Conv2d(in_channels, in_channels // down_rate, 1)
        self.la_conv2 = nn.Conv2d(in_channels // down_rate, num_convs, 1)
        self.reduction_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.norm = nn.GroupNorm(32, hidden_channels)
        nn.init.normal_(self.la_conv1.weight, std=0.001)
        nn.init.zeros_(self.la_conv1.bias)
        nn.init.normal_(self.la_conv2.weight, std=0.001)
        nn.init.zeros_(self.la_conv2.bias)
        nn.init.normal_(self.reduction_conv.weight, std=0.01)

    def forward(self, feat: Tensor, avg_feat: Tensor) -> Tensor:
        """Weight the stacked features by the layer attention and reduce them.

        Args:
            feat: stacked features of shape (B, num_convs * C, H, W).
            avg_feat: global average of feat, of shape (B, num_convs * C, 1, 1).

        Returns:
            Task feature of shape (B, C, H, W).
        """
        batch_size, _, height, width = feat.shape
        weight = self.la_conv2(F.relu(self.la_conv1(avg_feat))).sigmoid()

        # the attention scales the conv weight instead of the features, which saves memory
        conv_weight = weight.reshape(batch_size, 1, self.num_convs, 1) * self.reduction_conv.weight.reshape(1, self.hidden_channels, self.num_convs, self.hidden_channels)
        conv_weight = conv_weight.reshape(batch_size, self.hidden_channels, -1)
        feat = torch.bmm(conv_weight, feat.reshape(batch_size, -1, height * width)).reshape(batch_size, self.hidden_channels, height, width)
        return F.relu(self.norm(feat))


class TOODHead(Head):
    """Task-aligned head of TOOD, trained with ATSS for the first epochs and task alignment learning after."""

    def __init__(self, in_channels: list[int], in_strides: list[int], num_classes: int, hidden_channels: int = 256, num_convs: int = 6, num_deform_convs: int = 0, anchor_type: Literal["anchor_free", "anchor_based"] = "anchor_free", octave_base_scale: float = 8.0, initial_epochs: int = 4, atss_topk: int = 9, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, score_threshold: float = 0.05, pre_nms_topk: int = 1000, nms_threshold: float = 0.6, max_detections: int = 100) -> None:
        """Initialize the head.

        Args:
            in_channels: channels of each input feature map, all equal.
            in_strides: stride of each input feature map.
            num_classes: number of object classes.
            hidden_channels: channels of the stacked convs.
            num_convs: number of stacked convs.
            num_deform_convs: number of first stacked convs that are modulated deformable convs.
            anchor_type: whether the boxes are distances from the anchor centers or deltas from the anchors.
            octave_base_scale: square anchor size in strides.
            initial_epochs: number of first epochs trained with ATSS and the focal loss.
            atss_topk: number of closest anchors per level taken as ATSS candidates.
            topk: number of anchors with the highest alignment metric taken as candidates of each gt.
            alpha: power of the class score in the alignment metric.
            beta: power of the IoU in the alignment metric.
            score_threshold: minimum score to keep a box.
            pre_nms_topk: maximum boxes kept per level before NMS.
            nms_threshold: IoU threshold of NMS.
            max_detections: maximum detections per image.
        """
        super().__init__(in_channels, in_strides, num_classes)

        if len(set(in_channels)) != 1:
            raise ValueError(f"the head is shared over levels and needs equal channels, got {in_channels}")

        self.anchor_type = anchor_type
        self.octave_base_scale = octave_base_scale
        self.initial_epochs = initial_epochs
        self.atss_topk = atss_topk
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.score_threshold = score_threshold
        self.pre_nms_topk = pre_nms_topk
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.inter_convs = nn.ModuleList()

        for i in range(num_convs):
            channels = in_channels[0] if i == 0 else hidden_channels
            conv = ModulatedDeformConv(channels, hidden_channels, 4) if i < num_deform_convs else nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False)
            nn.init.normal_(conv.weight, std=0.01)
            self.inter_convs.append(nn.Sequential(conv, nn.GroupNorm(32, hidden_channels), nn.ReLU(inplace=True)))

        self.cls_decomp = TaskDecomposition(hidden_channels, num_convs, num_convs * 8)
        self.reg_decomp = TaskDecomposition(hidden_channels, num_convs, num_convs * 8)
        self.tood_cls = nn.Conv2d(hidden_channels, num_classes, 3, padding=1)
        self.tood_reg = nn.Conv2d(hidden_channels, 4, 3, padding=1)
        self.cls_prob_conv1 = nn.Conv2d(hidden_channels * num_convs, hidden_channels // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(hidden_channels // 4, 1, 3, padding=1)
        self.reg_offset_conv1 = nn.Conv2d(hidden_channels * num_convs, hidden_channels // 4, 1)
        self.reg_offset_conv2 = nn.Conv2d(hidden_channels // 4, 4 * 2, 3, padding=1)
        self.scales = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in in_strides])

        # the class biases start from a prior probability of 0.01
        bias = -math.log((1 - 0.01) / 0.01)

        for conv, std in [(self.tood_cls, 0.01), (self.tood_reg, 0.01), (self.cls_prob_conv1, 0.01), (self.cls_prob_conv2, 0.01), (self.reg_offset_conv1, 0.001), (self.reg_offset_conv2, 0.001)]:
            nn.init.normal_(conv.weight, std=std)
            nn.init.zeros_(conv.bias)

        nn.init.constant_(self.tood_cls.bias, bias)
        nn.init.constant_(self.cls_prob_conv2.bias, bias)

    def forward(self, features: list[Tensor], targets: list[Targets] | None = None) -> dict[str, Tensor]:
        """Predict the class scores and the boxes of every location.

        Args:
            features: neck feature maps, from high to low resolution.
            targets: unused.

        Returns:
            Class scores (B, N, C) as probabilities, xyxy boxes in strides (B, N, 4), xyxy anchors in pixels (N, 4),
            level of each anchor (N,), and the input size (width, height).
        """
        cls_scores, boxes, anchors, levels = [], [], [], []

        for level, (x, scale, stride) in enumerate(zip(features, self.scales, self.in_strides)):
            batch_size, _, height, width = x.shape
            level_anchors = self.create_anchors(height, width, stride, x.device)
            inter_feats = []

            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)

            feat = torch.cat(inter_feats, 1)
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)

            # the class score is the geometric mean of the class and the alignment probabilities
            cls_logits = self.tood_cls(cls_feat)
            cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat)))
            cls_score = (cls_logits.sigmoid() * cls_prob.sigmoid()).sqrt()

            reg = self.tood_reg(reg_feat)
            batch_anchors = level_anchors.repeat(batch_size, 1)

            if self.anchor_type == "anchor_free":
                distances = (reg.exp() * scale).float().permute(0, 2, 3, 1).reshape(-1, 4)
                centers = self.centers(batch_anchors) / stride
                level_boxes = torch.cat([centers - distances[:, :2], centers + distances[:, 2:]], 1)
            else:
                deltas = (reg * scale).float().permute(0, 2, 3, 1).reshape(-1, 4)
                level_boxes = self.apply_deltas(deltas, batch_anchors) / stride

            level_boxes = level_boxes.reshape(batch_size, height, width, 4).permute(0, 3, 1, 2)
            offset = self.reg_offset_conv2(F.relu(self.reg_offset_conv1(feat)))

            # each box side is sampled bilinearly at its own predicted offset
            sampling_weight = level_boxes.new_ones(4, 1, 1, 1)
            level_boxes = deform_conv2d(level_boxes.contiguous(), offset.contiguous(), sampling_weight)
            cls_scores.append(cls_score.flatten(2))
            boxes.append(level_boxes.flatten(2))
            anchors.append(level_anchors)
            levels.append(torch.full((len(level_anchors),), level, device=x.device))

        # exact only for inputs that are multiples of the strides
        image_size = features[0].new_tensor([features[0].shape[-1] * self.in_strides[0], features[0].shape[-2] * self.in_strides[0]])

        return {
            "cls_scores": torch.cat(cls_scores, 2).transpose(1, 2),
            "boxes": torch.cat(boxes, 2).transpose(1, 2),
            "anchors": torch.cat(anchors),
            "levels": torch.cat(levels),
            "image_size": image_size
        }

    def create_anchors(self, height: int, width: int, stride: int, device: torch.device) -> Tensor:
        """Create the square xyxy anchors of one level, centered at the top-left corner of each cell.

        Args:
            height: feature map height.
            width: feature map width.
            stride: feature map stride.
            device: device of the anchors.

        Returns:
            Anchors of shape (H * W, 4).
        """
        size = stride * self.octave_base_scale
        shift_y, shift_x = torch.meshgrid(torch.arange(height, device=device) * stride, torch.arange(width, device=device) * stride, indexing="ij")
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], -1).reshape(-1, 4).float()
        return shifts + shifts.new_tensor([-0.5 * size, -0.5 * size, 0.5 * size, 0.5 * size])

    @staticmethod
    def apply_deltas(deltas: Tensor, anchors: Tensor) -> Tensor:
        """Decode deltas with stds (0.1, 0.1, 0.2, 0.2) to xyxy boxes, clamping the log size ratios at log(1000 / 16)."""
        deltas = deltas * deltas.new_tensor([0.1, 0.1, 0.2, 0.2])
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
        max_ratio = abs(math.log(16 / 1000))
        centers = anchor_centers + anchor_sizes * deltas[:, :2]
        sizes = anchor_sizes * deltas[:, 2:].clamp(min=-max_ratio, max=max_ratio).exp()
        return torch.cat([centers - sizes * 0.5, centers + sizes * 0.5], 1)

    def decode(self, outputs: dict[str, Tensor]) -> list[Detections]:
        """Keep the top locations of each level by their best score, clip, and apply class-wise NMS above the threshold.

        Args:
            outputs: raw outputs of forward.

        Returns:
            Detections of each image.
        """
        levels = outputs["levels"]
        width, height = outputs["image_size"].tolist()
        scale = outputs["anchors"].new_tensor([width, height, width, height])
        detections = []

        for image_scores, image_boxes in zip(outputs["cls_scores"], outputs["boxes"]):
            kept_scores, kept_boxes = [], []

            for level, stride in enumerate(self.in_strides):
                level_scores = image_scores[levels == level]
                level_boxes = image_boxes[levels == level] * stride

                if 0 < self.pre_nms_topk < len(level_scores):
                    _, top = level_scores.max(1).values.topk(self.pre_nms_topk)
                    level_scores, level_boxes = level_scores[top], level_boxes[top]

                level_boxes[:, 0::2] = level_boxes[:, 0::2].clamp(min=0, max=width)
                level_boxes[:, 1::2] = level_boxes[:, 1::2].clamp(min=0, max=height)
                kept_scores.append(level_scores)
                kept_boxes.append(level_boxes)

            scores, boxes = torch.cat(kept_scores), torch.cat(kept_boxes)
            box_indices, labels = torch.nonzero(scores > self.score_threshold, as_tuple=True)
            boxes, scores = boxes[box_indices], scores[box_indices, labels]
            keep = batched_nms(boxes, scores, labels, self.nms_threshold)[:self.max_detections]
            detections.append(Detections(box_convert(boxes[keep], "xyxy", "cxcywh") / scale, scores[keep], labels[keep]))

        return detections

    def loss(self, outputs: dict[str, Tensor], targets: list[Targets], state: dict[str, int | float]) -> dict[str, Tensor]:
        """Compute the class loss and the weighted GIoU loss, with ATSS before initial_epochs and task alignment after.

        Args:
            outputs: raw outputs of forward.
            targets: ground truth of each image.
            state: training progress, with "epoch" (0-based).

        Returns:
            Losses, with the total under the key "loss".
        """
        cls_scores, boxes, anchors, levels = outputs["cls_scores"], outputs["boxes"], outputs["anchors"], outputs["levels"]
        width, height = outputs["image_size"].tolist()
        scale = anchors.new_tensor([width, height, width, height])
        strides = anchors.new_tensor(self.in_strides)[levels]
        num_level_anchors = torch.bincount(levels, minlength=len(self.in_strides)).tolist()
        initial = state["epoch"] < self.initial_epochs
        labels, gt_boxes, metrics = [], [], []

        for i, target in enumerate(targets):
            image_gt_boxes = box_convert(target.boxes, "cxcywh", "xyxy") * scale

            if initial:
                assigned = self.assign_atss(anchors, num_level_anchors, image_gt_boxes)
            else:
                assigned = self.assign_aligned(cls_scores[i], boxes[i] * strides[:, None], anchors, image_gt_boxes, target.labels)

            image_labels, image_boxes, image_metrics = self.targets(assigned, image_gt_boxes, target.labels, anchors, initial)
            labels.append(image_labels)
            gt_boxes.append(image_boxes)
            metrics.append(image_metrics)

        labels = torch.cat(labels)
        gt_boxes = torch.cat(gt_boxes)
        metrics = torch.cat(metrics)
        strides = strides.repeat(len(targets))
        pos = labels < self.num_classes
        probs = cls_scores.reshape(-1, self.num_classes)
        targets_onehot = F.one_hot(labels, self.num_classes + 1)[:, :-1].to(probs.dtype)

        # the normalizers are averaged over the processes of distributed training, only in training mode
        if initial:
            loss_cls = self.focal_loss(probs, targets_onehot).sum()
            cls_factor = pos.sum().to(probs.dtype)
        else:
            loss_cls = self.aligned_focal_loss(probs, targets_onehot * metrics[:, None]).sum()
            cls_factor = metrics.sum()

        cls_factor = (reduce_mean(cls_factor) if self.training else cls_factor).item()
        loss_cls = loss_cls / (cls_factor if cls_factor >= 1e-12 else 1)
        loss_box = boxes.sum() * 0
        box_weights = metrics[pos]

        if initial:
            box_weights = self.centerness(anchors.repeat(len(targets), 1)[pos], gt_boxes[pos])

        # every process joins the average, also without positives
        box_factor = box_weights.sum()
        box_factor = (reduce_mean(box_factor) if self.training else box_factor).item()

        if pos.any():
            # the GIoU loss weight is 2, computed in strides
            losses = self.giou_loss(boxes.reshape(-1, 4)[pos], gt_boxes[pos] / strides[pos, None]) * box_weights
            loss_box = 2.0 * losses.sum() / (box_factor if box_factor >= 1e-12 else 1)

        return {"loss": loss_cls + loss_box, "loss_cls": loss_cls, "loss_box": loss_box}

    def targets(self, assigned: tuple[Tensor, Tensor, Tensor], gt_boxes: Tensor, gt_labels: Tensor, anchors: Tensor, initial: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Build the targets of one image from an assignment.

        Args:
            assigned: assigned gt index (-1 for background), IoU and alignment metric of each anchor.
            gt_boxes: xyxy boxes in pixels of shape (G, 4).
            gt_labels: classes of shape (G,).
            anchors: xyxy anchors of shape (N, 4).
            initial: whether the assignment is ATSS, whose metric is one for the positives.

        Returns:
            Label of each anchor (num_classes for background), matched xyxy gt box, and the normalized alignment metric.
        """
        assigned_gt, ious, alignment = assigned
        pos = assigned_gt >= 0
        labels = gt_labels.new_full((len(anchors),), self.num_classes, dtype=torch.long)
        labels[pos] = gt_labels[assigned_gt[pos]].long()
        matched_boxes = torch.zeros_like(anchors)
        matched_boxes[pos] = gt_boxes[assigned_gt[pos]]
        metrics = pos.to(anchors.dtype)

        if initial:
            return labels, matched_boxes, metrics

        # the metric of each gt is scaled so that its best anchor gets the best IoU of that gt
        metrics = torch.zeros_like(metrics)

        for gt_index in torch.unique(assigned_gt[pos]):
            anchors_of_gt = assigned_gt == gt_index
            metrics[anchors_of_gt] = alignment[anchors_of_gt] / (alignment[anchors_of_gt].max() + 10e-8) * ious[anchors_of_gt].max()

        return labels, matched_boxes, metrics

    @staticmethod
    def pairwise_ious(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-6) -> Tensor:
        """IoU between every pair of xyxy boxes, with negative sides counted as zero and the union clamped at eps.

        Args:
            boxes1: boxes of shape (N, 4).
            boxes2: boxes of shape (G, 4).
            eps: minimum of the union.

        Returns:
            IoU of shape (N, G).
        """
        area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(1)
        area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(1)
        overlap = (torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:]) - torch.max(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0).prod(-1)
        union = torch.clamp(area1[:, None] + area2[None] - overlap, min=eps)
        return overlap / union

    @staticmethod
    def centers(boxes: Tensor) -> Tensor:
        """Centers of xyxy boxes."""
        return (boxes[:, :2] + boxes[:, 2:]) / 2.0

    @staticmethod
    def is_inside(points: Tensor, gt_boxes: Tensor) -> Tensor:
        """Whether the points are inside the gts by more than 0.01 pixels, of shape (N, G)."""
        deltas = torch.cat([points[:, None] - gt_boxes[None, :, :2], gt_boxes[None, :, 2:] - points[:, None]], -1)
        return deltas.min(-1).values > 0.01

    @torch.no_grad()
    def assign_atss(self, anchors: Tensor, num_level_anchors: list[int], gt_boxes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Assign the anchors of one image with ATSS, taking the closest anchors of each level as candidates.

        Args:
            anchors: xyxy anchors of shape (N, 4).
            num_level_anchors: number of anchors of each level.
            gt_boxes: xyxy boxes in pixels of shape (G, 4).

        Returns:
            Assigned gt index (-1 for background), IoU, and alignment metric (zero) of each anchor.
        """
        num_anchors = len(anchors)
        assigned_gt = anchors.new_full((num_anchors,), -1, dtype=torch.long)
        zeros = anchors.new_zeros(num_anchors)

        if len(gt_boxes) == 0:
            return assigned_gt, zeros, zeros

        ious = self.pairwise_ious(anchors, gt_boxes)
        distances = (self.centers(anchors)[:, None] - self.centers(gt_boxes)[None]).pow(2).sum(-1).sqrt()
        candidates = []
        start = 0

        for count in num_level_anchors:
            _, nearest = distances[start:start + count].topk(min(self.atss_topk, count), 0, largest=False)
            candidates.append(nearest + start)
            start += count

        candidates = torch.cat(candidates)
        gt_range = torch.arange(len(gt_boxes), device=anchors.device)
        candidate_ious = ious[candidates, gt_range]
        threshold = candidate_ious.mean(0) + candidate_ious.std(0)
        is_pos = (candidate_ious >= threshold) & self.is_inside(self.centers(anchors), gt_boxes)[candidates, gt_range]
        return self.pick(ious, candidates, is_pos, zeros)

    @torch.no_grad()
    def assign_aligned(self, cls_scores: Tensor, boxes: Tensor, anchors: Tensor, gt_boxes: Tensor, gt_labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Assign the anchors of one image by the alignment metric of the class score and the IoU of the predicted box.

        Args:
            cls_scores: class probabilities of shape (N, C).
            boxes: predicted xyxy boxes in pixels of shape (N, 4).
            anchors: xyxy anchors of shape (N, 4).
            gt_boxes: xyxy boxes in pixels of shape (G, 4).
            gt_labels: classes of shape (G,).

        Returns:
            Assigned gt index (-1 for background), IoU, and alignment metric of each anchor.
        """
        num_anchors = len(anchors)
        assigned_gt = anchors.new_full((num_anchors,), -1, dtype=torch.long)
        zeros = anchors.new_zeros(num_anchors)

        if len(gt_boxes) == 0:
            return assigned_gt, zeros, zeros

        ious = self.pairwise_ious(boxes, gt_boxes)
        alignment = cls_scores[:, gt_labels.long()] ** self.alpha * ious ** self.beta
        _, candidates = alignment.topk(self.topk, 0)
        gt_range = torch.arange(len(gt_boxes), device=anchors.device)
        is_pos = (alignment[candidates, gt_range] > 0) & self.is_inside(self.centers(anchors), gt_boxes)[candidates, gt_range]
        return self.pick(ious, candidates, is_pos, alignment)

    @staticmethod
    def pick(ious: Tensor, candidates: Tensor, is_pos: Tensor, alignment: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Assign each anchor to the positive candidate gt with the highest IoU.

        Args:
            ious: IoU of shape (N, G).
            candidates: candidate anchor indices of each gt, of shape (K, G).
            is_pos: whether each candidate is positive, of shape (K, G).
            alignment: alignment metric of shape (N, G), or zeros of shape (N,) for ATSS.

        Returns:
            Assigned gt index (-1 for background), IoU of the assigned gt, and alignment metric of the assigned gt.
        """
        num_anchors, num_gt = ious.shape
        gt_range = torch.arange(num_gt, device=ious.device).expand_as(candidates)
        masked = torch.full_like(ious, -1e8)
        masked[candidates[is_pos], gt_range[is_pos]] = ious[candidates[is_pos], gt_range[is_pos]]
        max_ious, argmax = masked.max(1)
        pos = max_ious != -1e8
        assigned_gt = torch.where(pos, argmax, -1)
        metrics = torch.zeros_like(max_ious)

        if alignment.dim() == 2:
            metrics[pos] = alignment[pos, argmax[pos]]

        return assigned_gt, max_ious, metrics

    @staticmethod
    def centerness(anchors: Tensor, gt_boxes: Tensor) -> Tensor:
        """Centerness of the anchor centers in the matched gts."""
        centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        left_top = centers - gt_boxes[:, :2]
        right_bottom = gt_boxes[:, 2:] - centers
        ratios = torch.min(left_top, right_bottom) / torch.max(left_top, right_bottom)
        return torch.sqrt(ratios[:, 0] * ratios[:, 1])

    @staticmethod
    def focal_loss(probs: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
        """Focal loss on probabilities."""
        alphas = torch.where(targets == 1, alpha, 1 - alpha)
        pt = torch.where(targets == 1, probs, 1 - probs)
        return alphas * torch.pow(1 - pt, gamma) * F.binary_cross_entropy(probs, targets, reduction="none")

    @staticmethod
    def aligned_focal_loss(probs: Tensor, soft_targets: Tensor, gamma: float = 2.0) -> Tensor:
        """Focal loss on probabilities with soft targets, modulated by the distance to the target."""
        return torch.pow((soft_targets - probs).abs(), gamma) * F.binary_cross_entropy(probs, soft_targets, reduction="none")

    @staticmethod
    def giou_loss(boxes: Tensor, gt_boxes: Tensor, eps: float = 1e-6) -> Tensor:
        """One minus the GIoU of aligned xyxy boxes, with negative sides counted as zero and the union and the enclosing area clamped at eps.

        Args:
            boxes: predicted boxes of shape (M, 4).
            gt_boxes: target boxes of shape (M, 4).
            eps: minimum of the union and the enclosing area.

        Returns:
            Loss of each box, of shape (M,).
        """
        area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
        gt_area = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=0).prod(1)
        overlap = (torch.min(boxes[:, 2:], gt_boxes[:, 2:]) - torch.max(boxes[:, :2], gt_boxes[:, :2])).clamp(min=0).prod(1)
        union = torch.clamp(area + gt_area - overlap, min=eps)
        enclosing = (torch.max(boxes[:, 2:], gt_boxes[:, 2:]) - torch.min(boxes[:, :2], gt_boxes[:, :2])).clamp(min=0).prod(1)
        enclosing = torch.clamp(enclosing, min=eps)
        ious = overlap / union
        return 1 - (ious - (enclosing - union) / enclosing)
