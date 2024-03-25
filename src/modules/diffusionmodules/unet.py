import math
from abc import abstractmethod
from functools import partial
from typing import Iterable, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import AttentionBlock
from modules.spatialattention import SpatialTransformer
from modules.utils import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


def convert_module_to_fp16(x):
    pass


def convert_module_to_fp32(x):
    pass


class TimestepBlock(nn.Module):
    """
    A module where forward() takes an additional timestep embedding.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        apply the forward pass of the module
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module where forward() takes an additional timestep embedding.
    """

    def forward(self, x, emb, context=None):
        for module in self:
            if isinstance(module, TimestepBlock):
                x = module(x, emb)
            elif isinstance(module, SpatialTransformer):
                x = module(x, context)
            else:
                x = module(x)
        return x


class Upsample(nn.Module):
    def __init(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.out_channels = out_channels or channels
        self.padding = padding

        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, channels, self.out_channels, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        out = self.emb_layers(emb).type(h.dtype)
        while len(out.shape) < len(h.shape):
            out = out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + out
            h = self.out_layers(h)
        return h + self.skip_connection(x)


class UNetModel(nn.Module):
    """
    A UNet model for image embedding.
    :param image_size: The size of the input image.
    :param in_channels: The number of input channels.
    :param model_channels: The number of channels in the model.
    :param out_channels: The number of output channels.
    :param num_res_blocks: The number of residual blocks in the model.
    :param attention_resolutions: A set of resolutions at which to apply attention.
    :param dropout: The dropout rate.
    :param channel_mult: The channel multiplier for each level of the model.
    :param conv_resample: Whether to use convolutional resampling.
    :param dims: The number of dimensions in the model.
    :param num_classes: The number of classes to predict.
    :param use_checkpoint: Whether to use checkpointing.
    :param use_fp16: Whether to use 16-bit floating point precision.
    :param num_heads: The number of attention heads.
    :param num_head_channels: The number of channels in the attention heads.
    :param num_heads_upsample: The number of attention heads to use for upsampling.
    :param use_scale_shift_norm: Whether to use scale and shift normalization.
    :param resblock_updown: Whether to use residual blocks for upsampling and downsampling.
    :param use_new_attention_order: Whether to use a new attention order.
    :param use_spatial_transformer: Whether to use a spatial transformer.
    :param transformer_depth: The depth of the transformer.
    :param context_dim: The dimension of the context.
    :param n_embed: The number of embeddings.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions: Set | List | Tuple,
        dropout: float = 0,
        channel_mult=(1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes=None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: int = 640,
        n_embed=None,
        legacy=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, "Spatial transformer requires context dim"

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Context dimension requires spatial transformer"
            from omegaconf.listconfig import ListConfig

            if isinstance(type(context_dim), ListConfig):
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Must specify num_heads or num_head_channels"

        if num_head_channels == -1:
            assert num_heads != -1, "Must specify num_head_channels or num_heads"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.class_embed = nn.Embedding(self.num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_heads == -1:
                        dim_head = ch // num_head_channels
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_channels.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_channels.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                in_ch = input_block_channels.pop()
                layers = [
                    ResBlock(
                        ch + in_ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_heads == -1:
                        dim_head = ch // num_head_channels
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims, out_channels=out_ch)
                    )
                    ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
            )

    def convert_to_fp16(self):
        """
        Convert the model to 16-bit floating point precision.
        """
        self.input_blocks.apply(convert_module_to_fp16)
        self.middle_block.apply(convert_module_to_fp16)
        self.output_blocks.apply(convert_module_to_fp16)

    def convert_to_fp32(self):
        """
        Convert the model to 32-bit floating point precision.
        """
        self.input_blocks.apply(convert_module_to_fp32)
        self.middle_block.apply(convert_module_to_fp32)
        self.output_blocks.apply(convert_module_to_fp32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (
            self.num_classes is not None
        ), "Must provide y if and only if model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y is not None
            class_emb = self.class_embed(y)
            emb = emb + class_emb

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)

        return self.out(h)
