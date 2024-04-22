import math

import torch
from torch import nn

from modules.utils import checkpoint, conv_nd, normalization, zero_module


class AttentionPool2D(nn.Module):
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        n_heads_channels: int,
        output_dim: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.positional_embeddings = nn.Parameter(
            torch.randn(embed_dim, spacial_dim**2 + 1 / embed_dim**0.5)
        )
        self.qvk_proj = conv_nd(embed_dim, embed_dim * 3, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // n_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spacial = x.shape
        x = x.reshape(b, c, -1)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embeddings[None, :, :].to(x.dtype)
        x = self.qvk_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), "channels must be divisible by num_head_channels"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(channels, channels * 3, 1)

        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *_spacial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return h


class QKVAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(bs * self.num_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs -> bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs -> bct", weight, v)
        return a.reshape(bs, -1, length)
