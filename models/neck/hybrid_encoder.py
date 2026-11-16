# Adapted from RT-DETR (Apache-2.0). See THIRD_PARTY_NOTICES.txt.
from collections.abc import Callable
import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation

from models.base import Neck


class RepVggBlock(nn.Module):
    """Sum of a 3x3 conv branch and a 1x1 conv branch with batch norm, followed by activation."""

    def __init__(self, channels: int, activation: Callable[..., nn.Module]) -> None:
        """Initialize the block.

        Args:
            channels: input and output channels.
            activation: activation layer.
        """
        super().__init__()
        self.conv1 = Conv2dNormActivation(channels, channels, 3, activation_layer=None)
        self.conv2 = Conv2dNormActivation(channels, channels, 1, activation_layer=None)
        self.act = activation(inplace=True)
        self.conv = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply the fused conv if it exists, otherwise both branches."""
        y = self.conv1(x) + self.conv2(x) if self.conv is None else self.conv(x)
        return self.act(y)

    @torch.no_grad()
    def fuse(self) -> None:
        """Fold both branches and their batch norms into a single 3x3 conv for inference."""
        channels = self.conv1[0].out_channels
        kernel = torch.zeros_like(self.conv1[0].weight)
        bias = torch.zeros(channels, device=kernel.device)

        for branch in [self.conv1, self.conv2]:
            conv, bn = branch[0], branch[1]
            scale = bn.weight / (bn.running_var + bn.eps).sqrt()
            pad = (3 - conv.kernel_size[0]) // 2
            kernel += F.pad(conv.weight * scale.reshape(-1, 1, 1, 1), [pad] * 4)
            bias += bn.bias - bn.running_mean * scale

        self.conv = nn.Conv2d(channels, channels, 3, padding=1).to(kernel.device)
        self.conv.weight.copy_(kernel)
        self.conv.bias.copy_(bias)


class CSPRepLayer(nn.Module):
    """CSP layer with RepVGG blocks, whose two branches are summed."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, expansion: float, activation: Callable[..., nn.Module]) -> None:
        """Initialize the layer.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            num_blocks: number of RepVGG blocks.
            expansion: ratio of the hidden channels to the output channels.
            activation: activation layer.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv2dNormActivation(in_channels, hidden_channels, 1, activation_layer=activation)
        self.conv2 = Conv2dNormActivation(in_channels, hidden_channels, 1, activation_layer=activation)
        self.blocks = nn.Sequential(*[RepVggBlock(hidden_channels, activation) for _ in range(num_blocks)])
        self.conv3 = nn.Identity() if hidden_channels == out_channels else Conv2dNormActivation(hidden_channels, out_channels, 1, activation_layer=activation)

    def forward(self, x: Tensor) -> Tensor:
        """Apply both branches, sum them and project."""
        main = self.blocks(self.conv1(x))
        short = self.conv2(x)
        return self.conv3(main + short)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer whose positional embedding is added to the queries and keys only."""

    def __init__(self, dim: int, num_heads: int, dim_feedforward: int, dropout: float, activation: Callable[..., nn.Module], normalize_before: bool) -> None:
        """Initialize the layer.

        Args:
            dim: token dimension.
            num_heads: number of attention heads.
            dim_feedforward: hidden dimension of the feedforward network.
            dropout: dropout probability.
            activation: activation layer of the feedforward network.
            normalize_before: whether to apply layer norm before each sublayer instead of after.
        """
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, src: Tensor, pos_embed: Tensor) -> Tensor:
        """Apply self-attention and the feedforward network.

        Args:
            src: tokens of shape (B, N, C).
            pos_embed: positional embedding of shape (1, N, C).

        Returns:
            Tokens of shape (B, N, C).
        """
        residual = src

        if self.normalize_before:
            src = self.norm1(src)

        query = key = src + pos_embed
        attended = self.self_attn(query, key, value=src, need_weights=False)[0]
        src = residual + self.dropout1(attended)

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src

        if self.normalize_before:
            src = self.norm2(src)

        hidden = self.dropout(self.activation(self.linear1(src)))
        src = residual + self.dropout2(self.linear2(hidden))

        if not self.normalize_before:
            src = self.norm2(src)

        return src


def build_position_embedding(width: int, height: int, dim: int, temperature: float) -> Tensor:
    """Build the 2D sine-cosine positional embedding of a row-major flattened grid.

    Args:
        width: grid width.
        height: grid height.
        dim: embedding dimension, divisible by 4.
        temperature: frequency temperature.

    Returns:
        Embedding of shape (1, height * width, dim).
    """
    if dim % 4 != 0:
        raise ValueError(f"embedding dimension must be divisible by 4, got {dim}")

    rows = torch.arange(height, dtype=torch.float32)
    cols = torch.arange(width, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
    omega = 1.0 / temperature ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    out_w = grid_w.flatten()[:, None] @ omega[None]
    out_h = grid_h.flatten()[:, None] @ omega[None]
    return torch.cat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], 1)[None]


class HybridEncoder(Neck):
    """Transformer encoder on selected levels followed by a CNN PAN, following the RT-DETR encoder."""

    def __init__(self, in_channels: list[int], in_strides: list[int], out_channels: int = 256, num_heads: int = 8, dim_feedforward: int = 1024, dropout: float = 0.0, encoder_activation: Callable[..., nn.Module] = nn.GELU, encoder_indices: tuple[int, ...] = (-1,), num_encoder_layers: int = 1, temperature: float = 10000.0, expansion: float = 1.0, num_blocks: int = 3, activation: Callable[..., nn.Module] = nn.SiLU, normalize_before: bool = False) -> None:
        """Initialize the neck.

        Args:
            in_channels: channels of each input feature map.
            in_strides: stride of each input feature map.
            out_channels: channels of every projected and output feature map, also the transformer dimension.
            num_heads: number of attention heads.
            dim_feedforward: hidden dimension of the transformer feedforward network.
            dropout: transformer dropout probability.
            encoder_activation: activation layer of the transformer feedforward network.
            encoder_indices: indices of the input levels that go through the transformer encoder.
            num_encoder_layers: number of transformer layers per encoded level.
            temperature: temperature of the positional embedding.
            expansion: ratio of the hidden channels to the output channels in each CSP layer.
            num_blocks: number of RepVGG blocks in each CSP layer.
            activation: activation layer of the CNN part.
            normalize_before: whether the transformer layers apply layer norm before each sublayer.
        """
        super().__init__(in_channels, in_strides)
        self.channels = out_channels
        self.encoder_indices = encoder_indices
        self.temperature = temperature
        self.eval_pos_embeds: dict[tuple[int, int], Tensor] = {}
        self.input_projs = nn.ModuleList([Conv2dNormActivation(channels, out_channels, 1, activation_layer=None) for channels in in_channels])

        # each encoder starts from copies of one layer, so all copies share the initial weights
        encoder_layer = TransformerEncoderLayer(out_channels, num_heads, dim_feedforward, dropout, encoder_activation, normalize_before)
        self.encoders = nn.ModuleList()

        for _ in encoder_indices:
            self.encoders.append(nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)]))

        self.lateral_convs = nn.ModuleList([Conv2dNormActivation(out_channels, out_channels, 1, activation_layer=activation) for _ in in_channels[1:]])
        self.top_down_layers = nn.ModuleList([CSPRepLayer(2 * out_channels, out_channels, num_blocks, expansion, activation) for _ in in_channels[1:]])
        self.downsample_convs = nn.ModuleList([Conv2dNormActivation(out_channels, out_channels, 3, 2, activation_layer=activation) for _ in in_channels[1:]])
        self.bottom_up_layers = nn.ModuleList([CSPRepLayer(2 * out_channels, out_channels, num_blocks, expansion, activation) for _ in in_channels[1:]])
        self.out_channels = [out_channels] * len(in_channels)
        self.out_strides = list(in_strides)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Encode the selected levels, then fuse all levels with the top-down and bottom-up paths.

        Args:
            features: backbone feature maps, from high to low resolution.

        Returns:
            Feature maps for the head, from high to low resolution.
        """
        projected = [proj(feature) for proj, feature in zip(self.input_projs, features)]

        for index, layers in zip(self.encoder_indices, self.encoders):
            batch_size, _, height, width = projected[index].shape
            tokens = projected[index].flatten(2).permute(0, 2, 1)

            # in eval mode the embedding of each feature size is computed once and reused
            if self.training:
                pos_embed = build_position_embedding(width, height, self.channels, self.temperature)
            else:
                if (height, width) not in self.eval_pos_embeds:
                    self.eval_pos_embeds[(height, width)] = build_position_embedding(width, height, self.channels, self.temperature)

                pos_embed = self.eval_pos_embeds[(height, width)]

            pos_embed = pos_embed.to(tokens.device)

            for layer in layers:
                tokens = layer(tokens, pos_embed)

            projected[index] = tokens.permute(0, 2, 1).reshape(batch_size, self.channels, height, width).contiguous()

        # the lateral conv output replaces the coarser map, and the bottom-up path concatenates it later
        top_down = [projected[-1]]

        for i, (lateral_conv, layer) in enumerate(zip(self.lateral_convs, self.top_down_layers)):
            reduced = lateral_conv(top_down[0])
            top_down[0] = reduced
            upsampled = F.interpolate(reduced, scale_factor=2.0, mode="nearest")
            top_down.insert(0, layer(torch.cat([upsampled, projected[-2 - i]], 1)))

        outputs = [top_down[0]]

        for downsample_conv, layer, lateral in zip(self.downsample_convs, self.bottom_up_layers, top_down[1:]):
            downsampled = downsample_conv(outputs[-1])
            outputs.append(layer(torch.cat([downsampled, lateral], 1)))

        return outputs
