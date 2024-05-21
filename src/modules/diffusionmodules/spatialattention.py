from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

from modules.utils import checkpoint, zero_module


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        self._glu = glu
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU(),
            )
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = einsum("b i j, b j k -> b i k", q, k)

        w_ = w_ / (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, " b i j -> b j i")
        h_ = torch.einsum("b i j, b j k -> b i k", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h, w=w)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, d_head=64, dropout=0.0, att_steps=1
    ):
        super().__init__()
        inner_dim = heads * d_head
        context_dim = default(context_dim, query_dim)

        self.scale = d_head**-0.5
        self.heads = heads
        self.att_steps = att_steps

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q, k, v = (
            self.to_q(x),
            self.to_k(default(context, x)),
            self.to_v(default(context, x)),
        )
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        limit = k.shape[0]
        att_steps = self.att_steps
        q_chunks = list(torch.tensor_split(q, limit // att_steps, dim=0))
        k_chunks = list(torch.tensor_split(k, limit // att_steps, dim=0))
        v_chunks = list(torch.tensor_split(v, limit // att_steps, dim=0))

        q_chunks.reverse()
        k_chunks.reverse()
        v_chunks.reverse()

        sim = torch.zeros(q.shape[0], q.shape[1], k.shape[2], device=q.device)
        del q, k, v

        for i in range(0, limit, att_steps):
            q_buffer = q_chunks.pop()
            k_buffer = k_chunks.pop()
            v_buffer = v_chunks.pop()
            sim_buffer = (
                einsum("b i d, b j d -> b i j", q_buffer, k_buffer) * self.scale
            )

            del q_buffer, k_buffer

            sim_buffer = sim_buffer.softmax(dim=-1)

            sim_buffer = einsum("b i j, b j d -> b i d", sim_buffer, v_buffer)
            del v_buffer
            sim[i : i + att_steps, :, :] = sim_buffer
            del sim_buffer

        sim = rearrange(sim, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(sim)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gatted_ff=True,
        use_checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, d_head=d_head, dropout=dropout
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gatted_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            d_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = use_checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    A transformer block for image-like inputs.
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout: float = 0.0,
        context_dim=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
