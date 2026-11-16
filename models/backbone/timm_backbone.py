import timm
from torch import Tensor

from models.base import Backbone


class TimmBackbone(Backbone):
    """Backbone that wraps a timm model."""

    def __init__(self, name: str, out_indices: tuple[int, ...] = (2, 3, 4), pretrained: bool = True) -> None:
        """Initialize the backbone.

        Args:
            name: timm model name.
            out_indices: indices of the feature levels to return.
            pretrained: whether to load the pretrained weights.
        """
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.out_channels = self.model.feature_info.channels()
        self.out_strides = self.model.feature_info.reduction()

    def forward(self, images: Tensor) -> list[Tensor]:
        """Extract the feature maps.

        Args:
            images: batch of images of shape (B, 3, H, W).

        Returns:
            Feature maps, from high to low resolution.
        """
        return self.model(images)
