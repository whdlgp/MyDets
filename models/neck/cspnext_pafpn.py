# Adapted from MMDetection and MMCV (Apache-2.0). See THIRD_PARTY_NOTICES.txt.
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation

from models.base import Neck


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise conv block followed by a pointwise conv block, each with batch norm and activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, activation_layer: Callable[..., nn.Module] = nn.SiLU) -> None:
        """Initialize the block.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            kernel_size: depthwise conv kernel size.
            stride: depthwise conv stride.
            activation_layer: activation layer.
        """
        super().__init__(
            Conv2dNormActivation(in_channels, in_channels, kernel_size, stride, groups=in_channels, activation_layer=activation_layer),
            Conv2dNormActivation(in_channels, out_channels, 1, activation_layer=activation_layer)
        )


class CSPNeXtBlock(nn.Module):
    """3x3 conv block followed by a 5x5 depthwise separable conv block, without identity."""

    def __init__(self, channels: int, depthwise: bool, activation: Callable[..., nn.Module]) -> None:
        """Initialize the block.

        Args:
            channels: input and output channels.
            depthwise: whether the 3x3 conv block is depthwise separable.
            activation: activation layer.
        """
        super().__init__()
        conv = DepthwiseSeparableConv if depthwise else Conv2dNormActivation
        self.conv1 = conv(channels, channels, 3, activation_layer=activation)
        self.conv2 = DepthwiseSeparableConv(channels, channels, 5, activation_layer=activation)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two conv blocks."""
        return self.conv2(self.conv1(x))


class CSPLayer(nn.Module):
    """CSP layer with CSPNeXt blocks: a block branch and a 1x1 branch, concatenated and fused."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, expand_ratio: float, depthwise: bool, activation: Callable[..., nn.Module]) -> None:
        """Initialize the layer.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            num_blocks: number of CSPNeXt blocks.
            expand_ratio: ratio of the hidden channels to the output channels.
            depthwise: whether the blocks use depthwise separable 3x3 convs.
            activation: activation layer.
        """
        super().__init__()
        hidden_channels = int(out_channels * expand_ratio)
        self.main_conv = Conv2dNormActivation(in_channels, hidden_channels, 1, activation_layer=activation)
        self.short_conv = Conv2dNormActivation(in_channels, hidden_channels, 1, activation_layer=activation)
        self.final_conv = Conv2dNormActivation(2 * hidden_channels, out_channels, 1, activation_layer=activation)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(hidden_channels, depthwise, activation) for _ in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        """Apply both branches and fuse them."""
        main = self.blocks(self.main_conv(x))
        short = self.short_conv(x)
        return self.final_conv(torch.cat([main, short], 1))


class CSPNeXtPAFPN(Neck):
    """PAN with CSPNeXt layers and output convs, following the RTMDet neck."""

    def __init__(self, in_channels: list[int], in_strides: list[int], out_channels: int = 256, num_blocks: int = 3, expand_ratio: float = 0.5, depthwise: bool = False, activation: Callable[..., nn.Module] = nn.SiLU) -> None:
        """Initialize the neck.

        Args:
            in_channels: channels of each input feature map.
            in_strides: stride of each input feature map.
            out_channels: channels of every output feature map.
            num_blocks: number of CSPNeXt blocks in each CSP layer.
            expand_ratio: ratio of the hidden channels to the output channels in each CSP layer.
            depthwise: whether to use depthwise separable convs for the 3x3 convs.
            activation: activation layer.
        """
        super().__init__(in_channels, in_strides)
        conv = DepthwiseSeparableConv if depthwise else Conv2dNormActivation
        self.reduce_convs = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        for high_channels, low_channels in zip(in_channels[:0:-1], in_channels[-2::-1]):
            self.reduce_convs.append(Conv2dNormActivation(high_channels, low_channels, 1, activation_layer=activation))
            self.top_down_layers.append(CSPLayer(2 * low_channels, low_channels, num_blocks, expand_ratio, depthwise, activation))

        for low_channels, high_channels in zip(in_channels[:-1], in_channels[1:]):
            self.downsample_convs.append(conv(low_channels, low_channels, 3, 2, activation_layer=activation))
            self.bottom_up_layers.append(CSPLayer(2 * low_channels, high_channels, num_blocks, expand_ratio, depthwise, activation))

        self.out_convs = nn.ModuleList([conv(channels, out_channels, 3, activation_layer=activation) for channels in in_channels])
        self.out_channels = [out_channels] * len(in_channels)
        self.out_strides = list(in_strides)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Fuse the feature maps with the top-down and bottom-up paths, then apply the output convs.

        Args:
            features: backbone feature maps, from high to low resolution.

        Returns:
            Feature maps for the head, from high to low resolution.
        """
        # the reduced map replaces the coarser map, and the bottom-up path concatenates it later
        top_down = [features[-1]]

        for i, (reduce_conv, layer) in enumerate(zip(self.reduce_convs, self.top_down_layers)):
            reduced = reduce_conv(top_down[0])
            top_down[0] = reduced
            upsampled = F.interpolate(reduced, scale_factor=2.0, mode="nearest")
            top_down.insert(0, layer(torch.cat([upsampled, features[-2 - i]], 1)))

        outputs = [top_down[0]]

        for downsample_conv, layer, lateral in zip(self.downsample_convs, self.bottom_up_layers, top_down[1:]):
            downsampled = downsample_conv(outputs[-1])
            outputs.append(layer(torch.cat([downsampled, lateral], 1)))

        return [out_conv(x) for out_conv, x in zip(self.out_convs, outputs)]
