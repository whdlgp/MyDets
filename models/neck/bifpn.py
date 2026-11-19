# Adapted from Google AutoML EfficientDet (Apache-2.0). See THIRD_PARTY_NOTICES.txt.
from typing import Literal
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.base import Neck


class Resample(nn.Module):
    """Resample a feature map to a target size, with a 1x1 conv and batch norm when the channels differ."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the resampler.

        Args:
            in_channels: input channels.
            out_channels: output channels.
        """
        super().__init__()
        self.conv = None

        if in_channels != out_channels:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))

    def forward(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Resample the feature map.

        Args:
            x: feature map of shape (B, C, H, W).
            target_size: (height, width) to resample to.

        Returns:
            Resampled feature map.
        """
        height, width = x.shape[-2:]
        target_height, target_width = target_size

        if self.conv is not None:
            x = self.conv(x)

        if height > target_height and width > target_width:
            return self.pool(x, int((height - 1) // target_height + 1), int((width - 1) // target_width + 1))

        if height > target_height or width > target_width:
            raise ValueError(f"cannot resample {height}x{width} to {target_height}x{target_width}")

        if height < target_height or width < target_width:
            x = F.interpolate(x, size=target_size, mode="nearest")

        return x

    def pool(self, x: Tensor, stride_height: int, stride_width: int) -> Tensor:
        """Max pool with a kernel one larger than the stride and TensorFlow "SAME" padding.

        Args:
            x: feature map of shape (B, C, H, W).
            stride_height: vertical stride.
            stride_width: horizontal stride.

        Returns:
            Pooled feature map of size ceil(H / stride_height) x ceil(W / stride_width).
        """
        # "SAME" pads more at the bottom and right, and the padding never wins the max
        height, width = x.shape[-2:]
        pad_height = (height + stride_height - 1) // stride_height * stride_height + 1 - height
        pad_width = (width + stride_width - 1) // stride_width * stride_width + 1 - width
        x = F.pad(x, [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2], value=-math.inf)
        return F.max_pool2d(x, (stride_height + 1, stride_width + 1), (stride_height, stride_width))


class Node(nn.Module):
    """BiFPN node: resample the inputs, fuse them with weights, then swish and separable conv with batch norm."""

    def __init__(self, level: int, input_ids: list[int], input_channels: list[int], channels: int, weight_method: Literal["fastattn", "attn", "sum"]) -> None:
        """Initialize the node.

        Args:
            level: index of the pyramid level the node outputs, from the highest resolution.
            input_ids: ids of the input nodes.
            input_channels: channels of each input node.
            channels: output channels.
            weight_method: fusion method, fast normalized ("fastattn"), softmax ("attn") or plain sum ("sum").
        """
        super().__init__()
        self.level = level
        self.input_ids = input_ids
        self.weight_method = weight_method
        self.resamples = nn.ModuleList([Resample(in_channels, channels) for in_channels in input_channels])
        self.weights = None if weight_method == "sum" else nn.Parameter(torch.ones(len(input_ids)))
        self.depthwise_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise_conv = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

    def forward(self, nodes: list[Tensor]) -> Tensor:
        """Compute the node output.

        Args:
            nodes: all node outputs so far, starting with the cell inputs from the highest resolution.

        Returns:
            Output feature map of the node.
        """
        target_size = tuple(nodes[self.level].shape[-2:])
        inputs = [resample(nodes[i], target_size) for i, resample in zip(self.input_ids, self.resamples)]

        if self.weight_method == "fastattn":
            weights = F.relu(self.weights)
            weights_sum = weights.sum()
            fused = sum(x * w / (weights_sum + 1e-4) for x, w in zip(inputs, weights))
        elif self.weight_method == "attn":
            weights = torch.softmax(self.weights, 0)
            fused = sum(x * w for x, w in zip(inputs, weights))
        else:
            fused = sum(inputs)

        x = F.silu(fused)
        x = self.pointwise_conv(self.depthwise_conv(x))
        return self.bn(x)


class BiFPNLayer(nn.Module):
    """One BiFPN layer: a top-down path followed by a bottom-up path with extra same-level inputs."""

    def __init__(self, in_channels: list[int], channels: int, weight_method: Literal["fastattn", "attn", "sum"]) -> None:
        """Initialize the layer.

        Args:
            in_channels: channels of each input level, from the highest resolution.
            channels: channels of every node.
            weight_method: fusion method of the nodes.
        """
        super().__init__()
        num_levels = len(in_channels)
        level_ids = [[i] for i in range(num_levels)]
        node_channels = list(in_channels)
        node_specs = []

        for level in range(num_levels - 2, -1, -1):
            node_specs.append((level, [level_ids[level][-1], level_ids[level + 1][-1]]))
            level_ids[level].append(num_levels + len(node_specs) - 1)

        for level in range(1, num_levels):
            node_specs.append((level, level_ids[level] + [level_ids[level - 1][-1]]))
            level_ids[level].append(num_levels + len(node_specs) - 1)

        self.nodes = nn.ModuleList()

        for level, input_ids in node_specs:
            self.nodes.append(Node(level, input_ids, [node_channels[i] for i in input_ids], channels, weight_method))
            node_channels.append(channels)

        self.output_ids = [ids[-1] for ids in level_ids]

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Apply the nodes in order.

        Args:
            features: input feature maps, from high to low resolution.

        Returns:
            Output feature maps, from high to low resolution.
        """
        nodes = list(features)

        for node in self.nodes:
            nodes.append(node(nodes))

        return [nodes[i] for i in self.output_ids]


class BiFPN(Neck):
    """Weighted bi-directional feature pyramid network, following the EfficientDet neck."""

    def __init__(self, in_channels: list[int], in_strides: list[int], out_channels: int = 64, num_layers: int = 3, extra_levels: int = 2, weight_method: Literal["fastattn", "attn", "sum"] = "fastattn") -> None:
        """Initialize the neck.

        Args:
            in_channels: channels of each input feature map.
            in_strides: stride of each input feature map.
            out_channels: channels of every BiFPN node and output.
            num_layers: number of stacked BiFPN layers.
            extra_levels: number of coarser levels added by downsampling the coarsest input.
            weight_method: fusion method, fast normalized ("fastattn"), softmax ("attn") or plain sum ("sum").
        """
        super().__init__(in_channels, in_strides)
        self.extra_resamples = nn.ModuleList([Resample(in_channels[-1] if i == 0 else out_channels, out_channels) for i in range(extra_levels)])
        level_channels = list(in_channels) + [out_channels] * extra_levels
        self.out_channels = [out_channels] * len(level_channels)
        self.out_strides = list(in_strides) + [in_strides[-1] * 2 ** (i + 1) for i in range(extra_levels)]
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(BiFPNLayer(level_channels, out_channels, weight_method))
            level_channels = self.out_channels

        # glorot uniform kernels and zero biases, the Keras defaults
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Add the extra levels and apply the BiFPN layers.

        Args:
            features: backbone feature maps, from high to low resolution.

        Returns:
            Feature maps for the head, from high to low resolution.
        """
        features = list(features)

        for resample in self.extra_resamples:
            height, width = features[-1].shape[-2:]
            features.append(resample(features[-1], ((height + 1) // 2, (width + 1) // 2)))

        for layer in self.layers:
            features = layer(features)

        return features
