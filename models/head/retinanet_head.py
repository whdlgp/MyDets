# Adapted from Detectron2 (Apache-2.0). See THIRD_PARTY_NOTICES.txt.
from typing import Literal
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import batched_nms, box_convert, box_iou, complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss, sigmoid_focal_loss

from models.base import Detections, Head, Targets


class LevelBatchNorm(nn.Module):
    """Batch norm with statistics per level and an affine transform shared over levels."""

    def __init__(self, channels: int, num_levels: int) -> None:
        """Initialize the norm.

        Args:
            channels: number of channels.
            num_levels: number of feature levels.
        """
        super().__init__()
        self.norms = nn.ModuleList([nn.BatchNorm2d(channels, affine=False) for _ in range(num_levels)])
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor, level: int) -> Tensor:
        """Normalize with the statistics of the level and apply the shared affine transform."""
        return self.norms[level](x) * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)


class RetinaNetHead(Head):
    """RetinaNet head: shared class and box subnets over all levels, anchors, focal loss and NMS."""

    def __init__(self, in_channels: list[int], in_strides: list[int], num_classes: int, num_convs: int = 4, norm: Literal["none", "bn", "gn"] = "none", prior_prob: float = 0.01, octave_base_scale: float = 4.0, scales_per_octave: int = 3, aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0), iou_thresholds: tuple[float, float] = (0.4, 0.5), focal_alpha: float = 0.25, focal_gamma: float = 2.0, box_loss: Literal["smooth_l1", "giou", "diou", "ciou"] = "smooth_l1", smooth_l1_beta: float = 0.0, score_threshold: float = 0.05, pre_nms_topk: int = 1000, nms_threshold: float = 0.5, max_detections: int = 100) -> None:
        """Initialize the head.

        Args:
            in_channels: channels of each input feature map, all equal.
            in_strides: stride of each input feature map.
            num_classes: number of object classes.
            num_convs: number of 3x3 convs in each subnet.
            norm: norm after each subnet conv: none, batch norm with statistics per level, or group norm with 32 groups.
            prior_prob: initial foreground probability of the class predictions.
            octave_base_scale: anchor size of the first scale in strides.
            scales_per_octave: number of anchor scales per octave.
            aspect_ratios: anchor height to width ratios.
            iou_thresholds: IoU below the first is background, at or above the second is foreground, ignored between.
            focal_alpha: alpha of the focal loss.
            focal_gamma: gamma of the focal loss.
            box_loss: box regression loss.
            smooth_l1_beta: beta of the smooth L1 loss, 0 for L1.
            score_threshold: minimum score to keep a box.
            pre_nms_topk: maximum boxes kept per level before NMS.
            nms_threshold: IoU threshold of NMS.
            max_detections: maximum detections per image.
        """
        super().__init__(in_channels, in_strides, num_classes)

        if len(set(in_channels)) != 1:
            raise ValueError(f"the subnets are shared over levels and need equal channels, got {in_channels}")

        self.iou_thresholds = iou_thresholds
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.box_loss = box_loss
        self.smooth_l1_beta = smooth_l1_beta
        self.score_threshold = score_threshold
        self.pre_nms_topk = pre_nms_topk
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.cell_anchors = [self.create_cell_anchors(octave_base_scale * stride, scales_per_octave, aspect_ratios) for stride in in_strides]
        self.num_anchors = scales_per_octave * len(aspect_ratios)
        self.loss_normalizer = 100.0
        channels = in_channels[0]
        num_levels = len(in_strides)
        self.cls_subnet = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(num_convs)])
        self.bbox_subnet = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(num_convs)])
        self.cls_norms = nn.ModuleList([self.create_norm(norm, channels, num_levels) for _ in range(num_convs)])
        self.bbox_norms = nn.ModuleList([self.create_norm(norm, channels, num_levels) for _ in range(num_convs)])
        self.cls_score = nn.Conv2d(channels, self.num_anchors * num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(channels, self.num_anchors * 4, 3, padding=1)

        for conv in [*self.cls_subnet, *self.bbox_subnet, self.cls_score, self.bbox_pred]:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, 0)

        nn.init.constant_(self.cls_score.bias, -math.log((1 - prior_prob) / prior_prob))

    @staticmethod
    def create_norm(norm: Literal["none", "bn", "gn"], channels: int, num_levels: int) -> nn.Module:
        """Create the norm of one subnet conv.

        Args:
            norm: norm type.
            channels: conv output channels.
            num_levels: number of feature levels.

        Returns:
            Batch norm with statistics per level, or group norm or identity shared over levels.
        """
        if norm == "bn":
            return LevelBatchNorm(channels, num_levels)

        return nn.GroupNorm(32, channels) if norm == "gn" else nn.Identity()

    @staticmethod
    def create_cell_anchors(base_size: float, scales_per_octave: int, aspect_ratios: tuple[float, ...]) -> Tensor:
        """Create the xyxy anchors of one location centered at zero, sizes in the outer loop.

        Args:
            base_size: size of the first scale in pixels.
            scales_per_octave: number of scales per octave.
            aspect_ratios: height to width ratios.

        Returns:
            Anchors of shape (A, 4).
        """
        anchors = []

        for i in range(scales_per_octave):
            area = (base_size * 2 ** (i / scales_per_octave)) ** 2.0

            for ratio in aspect_ratios:
                width = math.sqrt(area / ratio)
                height = ratio * width
                anchors.append([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        return torch.tensor(anchors)

    def forward(self, features: list[Tensor], targets: list[Targets] | None = None) -> dict[str, Tensor]:
        """Predict the raw outputs of every anchor.

        Args:
            features: neck feature maps, from high to low resolution.
            targets: unused.

        Returns:
            Raw class logits (B, N, C), box deltas (B, N, 4), xyxy anchors in pixels (N, 4),
            level of each anchor (N,), and the input size (width, height).
        """
        cls_logits, box_deltas, anchors, levels = [], [], [], []

        for level, (x, stride, cell_anchors) in enumerate(zip(features, self.in_strides, self.cell_anchors)):
            cls_feat = x
            box_feat = x

            for conv, norm in zip(self.cls_subnet, self.cls_norms):
                cls_feat = F.relu(self.normalize(norm, conv(cls_feat), level))

            for conv, norm in zip(self.bbox_subnet, self.bbox_norms):
                box_feat = F.relu(self.normalize(norm, conv(box_feat), level))

            batch_size, _, height, width = x.shape
            logits = self.cls_score(cls_feat).view(batch_size, self.num_anchors, self.num_classes, height, width)
            deltas = self.bbox_pred(box_feat).view(batch_size, self.num_anchors, 4, height, width)
            cls_logits.append(logits.permute(0, 3, 4, 1, 2).reshape(batch_size, -1, self.num_classes))
            box_deltas.append(deltas.permute(0, 3, 4, 1, 2).reshape(batch_size, -1, 4))

            shifts_x = torch.arange(0, width * stride, step=stride, dtype=torch.float32, device=x.device)
            shifts_y = torch.arange(0, height * stride, step=stride, dtype=torch.float32, device=x.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], -1).reshape(-1, 1, 4)
            level_anchors = (shifts + cell_anchors.to(x.device)).reshape(-1, 4)
            anchors.append(level_anchors)
            levels.append(torch.full((len(level_anchors),), level, device=x.device))

        # exact only for inputs that are multiples of the strides
        image_size = features[0].new_tensor([features[0].shape[-1] * self.in_strides[0], features[0].shape[-2] * self.in_strides[0]])
        return {"cls_logits": torch.cat(cls_logits, 1), "box_deltas": torch.cat(box_deltas, 1), "anchors": torch.cat(anchors), "levels": torch.cat(levels), "image_size": image_size}

    @staticmethod
    def normalize(norm: nn.Module, x: Tensor, level: int) -> Tensor:
        """Apply a subnet norm, passing the level to the batch norm with statistics per level."""
        return norm(x, level) if isinstance(norm, LevelBatchNorm) else norm(x)

    def decode(self, outputs: dict[str, Tensor]) -> list[Detections]:
        """Keep the top scoring anchor-class pairs of each level above the threshold and apply class-wise NMS.

        Args:
            outputs: raw outputs of forward.

        Returns:
            Detections of each image.
        """
        anchors, levels = outputs["anchors"], outputs["levels"]
        width, height = outputs["image_size"].tolist()
        scale = anchors.new_tensor([width, height, width, height])
        detections = []

        for logits, deltas in zip(outputs["cls_logits"], outputs["box_deltas"]):
            boxes, scores, labels = [], [], []

            for level in range(len(self.in_strides)):
                in_level = levels == level
                level_scores = logits[in_level].sigmoid()
                keep = level_scores > self.score_threshold
                level_scores = level_scores[keep]
                candidates = torch.nonzero(keep)
                level_scores, order = level_scores.topk(min(len(candidates), self.pre_nms_topk))
                anchor_indices, class_indices = candidates[order].unbind(1)
                boxes.append(self.apply_deltas(deltas[in_level][anchor_indices], anchors[in_level][anchor_indices]))
                scores.append(level_scores)
                labels.append(class_indices)

            boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
            keep = batched_nms(boxes, scores, labels, self.nms_threshold)[:self.max_detections]
            detections.append(Detections(box_convert(boxes[keep], "xyxy", "cxcywh") / scale, scores[keep], labels[keep]))

        return detections

    def loss(self, outputs: dict[str, Tensor], targets: list[Targets], state: dict[str, int | float]) -> dict[str, Tensor]:
        """Label the anchors by IoU and compute the focal loss and the box loss, normalized by an EMA of the positives.

        Args:
            outputs: raw outputs of forward.
            targets: ground truth of each image.
            state: unused.

        Returns:
            Losses, with the total under the key "loss".
        """
        cls_logits, box_deltas, anchors = outputs["cls_logits"], outputs["box_deltas"], outputs["anchors"]
        width, height = outputs["image_size"].tolist()
        scale = anchors.new_tensor([width, height, width, height])
        gt_labels, gt_boxes = [], []

        for target in targets:
            labels, boxes = self.label(box_convert(target.boxes, "cxcywh", "xyxy") * scale, target.labels, anchors)
            gt_labels.append(labels)
            gt_boxes.append(boxes)

        gt_labels = torch.stack(gt_labels)
        gt_boxes = torch.stack(gt_boxes)
        valid_mask = gt_labels >= 0
        pos_mask = valid_mask & (gt_labels != self.num_classes)
        num_pos = pos_mask.sum().item()

        # the normalizer is an EMA of the positives, starting from 100 with momentum 0.9
        momentum = 0.9
        self.loss_normalizer = self.loss_normalizer * momentum + max(num_pos, 1) * (1 - momentum)

        cls_targets = F.one_hot(gt_labels[valid_mask], self.num_classes + 1)[:, :-1].to(cls_logits.dtype)
        loss_cls = sigmoid_focal_loss(cls_logits[valid_mask], cls_targets, self.focal_alpha, self.focal_gamma, "sum")

        if self.box_loss == "smooth_l1":
            gt_deltas = torch.stack([self.get_deltas(anchors, boxes) for boxes in gt_boxes])
            loss_box = self.smooth_l1_loss(box_deltas[pos_mask], gt_deltas[pos_mask])
        else:
            boxes = torch.stack([self.apply_deltas(deltas, anchors) for deltas in box_deltas])
            box_iou_loss = {"giou": generalized_box_iou_loss, "diou": distance_box_iou_loss, "ciou": complete_box_iou_loss}[self.box_loss]
            loss_box = box_iou_loss(boxes[pos_mask], gt_boxes[pos_mask], "sum")

        loss_cls = loss_cls / self.loss_normalizer
        loss_box = loss_box / self.loss_normalizer
        return {"loss": loss_cls + loss_box, "loss_cls": loss_cls, "loss_box": loss_box}

    def smooth_l1_loss(self, deltas: Tensor, gt_deltas: Tensor) -> Tensor:
        """Sum of the smooth L1 loss, which is the L1 loss when beta is close to zero."""
        if self.smooth_l1_beta < 1e-5:
            return (deltas - gt_deltas).abs().sum()

        return F.smooth_l1_loss(deltas, gt_deltas, reduction="sum", beta=self.smooth_l1_beta)

    @torch.no_grad()
    def label(self, gt_boxes: Tensor, gt_labels: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor]:
        """Label the anchors of one image by their best IoU with the gts.

        Args:
            gt_boxes: xyxy boxes in pixels of shape (G, 4).
            gt_labels: classes of shape (G,).
            anchors: xyxy anchors of shape (N, 4).

        Returns:
            Label of each anchor, -1 for ignored and num_classes for background, and the matched xyxy gt box.
        """
        if len(gt_boxes) == 0:
            return gt_labels.new_full((len(anchors),), self.num_classes, dtype=torch.long), torch.zeros_like(anchors)

        ious = box_iou(gt_boxes, anchors)
        matched_ious, matches = ious.max(0)
        low, high = self.iou_thresholds
        match_labels = torch.ones_like(matches)
        match_labels[matched_ious < low] = 0
        match_labels[(matched_ious >= low) & (matched_ious < high)] = -1

        # every gt keeps the anchors with its highest IoU, ties included, even below the threshold
        highest = ious.max(1, keepdim=True).values
        _, best_anchors = torch.nonzero(ious == highest, as_tuple=True)
        match_labels[best_anchors] = 1

        labels = gt_labels[matches].long()
        labels[match_labels == 0] = self.num_classes
        labels[match_labels == -1] = -1
        return labels, gt_boxes[matches]

    def get_deltas(self, anchors: Tensor, boxes: Tensor) -> Tensor:
        """Encode xyxy boxes as center offsets and log size ratios to the xyxy anchors."""
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        anchor_centers = anchors[:, :2] + 0.5 * anchor_sizes
        sizes = boxes[:, 2:] - boxes[:, :2]
        centers = boxes[:, :2] + 0.5 * sizes
        return torch.cat([(centers - anchor_centers) / anchor_sizes, torch.log(sizes / anchor_sizes)], 1)

    def apply_deltas(self, deltas: Tensor, anchors: Tensor) -> Tensor:
        """Decode the deltas to xyxy boxes, clamping the log size ratios at log(1000 / 16)."""
        deltas = deltas.float()
        anchors = anchors.to(deltas.dtype)
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        anchor_centers = anchors[:, :2] + 0.5 * anchor_sizes
        log_ratios = torch.clamp(deltas[:, 2:], max=math.log(1000.0 / 16))
        centers = deltas[:, :2] * anchor_sizes + anchor_centers
        sizes = torch.exp(log_ratios) * anchor_sizes
        return torch.cat([centers - 0.5 * sizes, centers + 0.5 * sizes], 1)
