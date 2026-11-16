from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Self

from torch import Tensor, nn
from torchvision.ops import box_convert


@dataclass
class Targets:
    """Ground truth of one image, with normalized cxcywh boxes."""
    boxes: Tensor
    labels: Tensor
    extras: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class Detections:
    """Decoded result of one image, with normalized cxcywh boxes."""
    boxes: Tensor
    scores: Tensor
    labels: Tensor

    def convert(self, box_format: Literal["xyxy", "xywh", "cxcywh"] = "cxcywh", image_size: tuple[int, int] | None = None) -> Tensor:
        """Convert the boxes to the given format.

        Args:
            box_format: output box format.
            image_size: (width, height) to scale the boxes to pixels, or None to keep them normalized.

        Returns:
            Boxes of shape (N, 4) in the given format.
        """
        boxes = box_convert(self.boxes, "cxcywh", box_format)

        if image_size is not None:
            width, height = image_size
            boxes = boxes * boxes.new_tensor([width, height, width, height])

        return boxes


class Backbone(nn.Module, ABC):
    """Feature extractor that returns multi-scale feature maps."""

    out_channels: list[int]
    out_strides: list[int]

    def __init__(self) -> None:
        """Initialize the backbone as not frozen."""
        super().__init__()
        self.frozen = False

    @abstractmethod
    def forward(self, images: Tensor) -> list[Tensor]:
        """Extract the feature maps.

        Args:
            images: batch of images of shape (B, 3, H, W).

        Returns:
            Feature maps, from high to low resolution.
        """

    def freeze(self, frozen: bool = True) -> None:
        """Freeze or unfreeze everything the training can change.

        Args:
            frozen: whether to freeze the backbone.
        """
        self.frozen = frozen
        self.requires_grad_(not frozen)
        self.train(not frozen)

    def train(self, mode: bool = True) -> Self:
        """Set the training mode, keeping a frozen backbone in eval mode."""
        return super().train(mode and not self.frozen)


class Neck(nn.Module, ABC):
    """Module between the backbone and the head."""

    out_channels: list[int]
    out_strides: list[int]

    def __init__(self, in_channels: list[int], in_strides: list[int]) -> None:
        """Initialize the neck.

        Args:
            in_channels: channels of each input feature map.
            in_strides: stride of each input feature map.
        """
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides

    @abstractmethod
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Transform the backbone feature maps.

        Args:
            features: backbone feature maps, from high to low resolution.

        Returns:
            Feature maps for the head, from high to low resolution.
        """


class Head(nn.Module, ABC):
    """Detection head that owns its decoding and loss."""

    def __init__(self, in_channels: list[int], in_strides: list[int], num_classes: int) -> None:
        """Initialize the head.

        Args:
            in_channels: channels of each input feature map.
            in_strides: stride of each input feature map.
            num_classes: number of object classes.
        """
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, features: list[Tensor], targets: list[Targets] | None = None) -> dict[str, Tensor]:
        """Predict the raw outputs.

        Args:
            features: neck feature maps, from high to low resolution.
            targets: ground truth, only for heads that use it in training.

        Returns:
            Raw outputs, interpreted only by this head.
        """

    @abstractmethod
    def decode(self, outputs: dict[str, Tensor]) -> list[Detections]:
        """Decode the raw outputs, including the post-processing of the head.

        Args:
            outputs: raw outputs of forward.

        Returns:
            Detections of each image.
        """

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], targets: list[Targets], state: dict[str, int | float]) -> dict[str, Tensor]:
        """Compute the losses.

        Args:
            outputs: raw outputs of forward.
            targets: ground truth of each image.
            state: training progress such as the epoch, ignored by heads that do not need it.

        Returns:
            Losses, with the total under the key "loss".
        """


class Detector(nn.Module):
    """Detector made of a backbone, a neck and a head."""

    def __init__(self, backbone: Backbone, neck: Neck, head: Head) -> None:
        """Initialize the detector.

        Args:
            backbone: feature extractor.
            neck: module between the backbone and the head.
            head: detection head, used directly for decode and loss.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, images: Tensor, targets: list[Targets] | None = None) -> dict[str, Tensor]:
        """Predict the raw outputs of the head.

        Args:
            images: batch of images of shape (B, 3, H, W).
            targets: ground truth, only for heads that use it in training.

        Returns:
            Raw outputs of the head.
        """
        features = self.backbone(images)
        features = self.neck(features)
        return self.head(features, targets)
