# Adapted from PANet (MIT License). See THIRD_PARTY_NOTICES.txt.
from functools import partial

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from models.base import Neck


class PAN(Neck):
    """Path aggregation network: a feature pyramid network followed by a bottom-up path."""

    def __init__(self, in_channels: list[int], in_strides: list[int], out_channels: int = 256, group_norm: bool = True, extra_levels: int = 0) -> None:
        """Initialize the neck.

        Args:
            in_channels: channels of each input feature map.
            in_strides: stride of each input feature map.
            out_channels: channels of every output feature map.
            group_norm: whether every conv is followed by group norm, and the bottom-up convs also by relu.
            extra_levels: number of coarser levels added by stride 2 subsampling of the coarsest output, 0 or 1.
        """
        super().__init__(in_channels, in_strides)

        if extra_levels not in (0, 1):
            raise ValueError(f"extra_levels must be 0 or 1, got {extra_levels}")

        norm_layer = partial(nn.GroupNorm, 32) if group_norm else None
        self.fpn = FeaturePyramidNetwork(in_channels, out_channels, norm_layer=norm_layer)
        self.downsample_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()

        for _ in in_channels[1:]:
            downsample_conv = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
            fuse_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            nn.init.kaiming_uniform_(downsample_conv.weight, a=1)
            nn.init.kaiming_uniform_(fuse_conv.weight, a=1)
            nn.init.zeros_(downsample_conv.bias)
            nn.init.zeros_(fuse_conv.bias)

            if group_norm:
                downsample_conv = nn.Sequential(downsample_conv, nn.GroupNorm(32, out_channels), nn.ReLU(inplace=True))
                fuse_conv = nn.Sequential(fuse_conv, nn.GroupNorm(32, out_channels), nn.ReLU(inplace=True))

            self.downsample_convs.append(downsample_conv)
            self.fuse_convs.append(fuse_conv)

        self.extra_levels = extra_levels
        self.out_strides = list(in_strides) + [in_strides[-1] * 2] * extra_levels

        self.out_channels = [out_channels] * len(self.out_strides)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Build the feature pyramid and augment it with the bottom-up path.

        Args:
            features: backbone feature maps, from high to low resolution.

        Returns:
            Feature maps for the head, from high to low resolution.
        """
        inputs = {str(i): feature for i, feature in enumerate(features)}
        pyramid = list(self.fpn(inputs).values())
        outputs = [pyramid[0]]

        for downsample_conv, fuse_conv, lateral in zip(self.downsample_convs, self.fuse_convs, pyramid[1:]):
            downsampled = downsample_conv(outputs[-1])
            outputs.append(fuse_conv(downsampled + lateral))

        if self.extra_levels:
            outputs.append(F.max_pool2d(outputs[-1], kernel_size=1, stride=2))

        return outputs
