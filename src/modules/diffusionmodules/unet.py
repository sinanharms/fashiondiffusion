from abc import abstractmethod
from typing import Any, List, Set, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.diffusionmodules.attention import AttentionBlock
from modules.diffusionmodules.spatialattention import SpatialTransformer
from modules.utils import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    default,
    get_device,
    instantiate_from_config,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


def convert_module_to_fp16(x):
    """
    Convert a module to 16-bit floating point precision.
    """
    if isinstance(x, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        x.weight.data = x.weight.data.half()
        if x.bias is not None:
            x.bias.data = x.bias.data.half()


def convert_module_to_fp32(x):
    """
    Convert a module to 32-bit floating point precision.
    """
    if isinstance(x, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        x.weight.data = x.weight.data.float()
        if x.bias is not None:
            x.bias.data = x.bias.data.float()


class UNetModel(nn.Module):
    """
    A UNet .model for image embedding.
    :param image_size: The size of the input image.
    :param in_channels: The number of input channels.
    :param model_channels: The number of channels in the .model.
    :param out_channels: The number of output channels.
    :param num_res_blocks: The number of residual blocks in the .model.
    :param attention_resolutions: A set of resolutions at which to apply attention.
    :param dropout: The dropout rate.
    :param channel_mult: The channel multiplier for each level of the .model.
    :param conv_resample: Whether to use convolutional resampling.
    :param dims: The number of dimensions in the .model.
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
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
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

        ch = input_ch = int(model_channels * channel_mult[0])
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_channels = [ch]
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
                ch = int(mult * model_channels)
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
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                )
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
                zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
            )

    def convert_to_fp16(self):
        """
        Convert the .model to 16-bit floating point precision.
        """
        self.input_blocks.apply(convert_module_to_fp16)
        self.middle_block.apply(convert_module_to_fp16)
        self.output_blocks.apply(convert_module_to_fp16)

    def convert_to_fp32(self):
        """
        Convert the .model to 32-bit floating point precision.
        """
        self.input_blocks.apply(convert_module_to_fp32)
        self.middle_block.apply(convert_module_to_fp32)
        self.output_blocks.apply(convert_module_to_fp32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (
            self.num_classes is not None
        ), "Must provide y if and only if .model is class-conditional"
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


class UNet(pl.LightningModule):
    def __init__(
        self,
        unet_encoder_config,
        unet_decoder_config,
        num_timestep_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        unet_bs=1,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timestep_cond = default(num_timestep_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timestep_cond <= kwargs["timesteps"]
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        self._device = get_device()
        self.unet_encoder_config = unet_encoder_config
        self.unet_decoder_config = unet_decoder_config
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.encode = UNetEncoderWrapper(unet_encoder_config)
        self.decode = UNetDecoderWrapper(unet_decoder_config)
        self.encode.eval()
        self.decode.eval()
        self.turbo = False
        self.unet_bs = unet_bs
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timestep_cond)
        ).long()
        self.cond_ids[: self.num_timestep_cond] = ids


class UNetEncoderWrapper(pl.LightningModule):
    def __init__(self, unet_encoder_config):
        super().__init__()
        self.unet_encoder = instantiate_from_config(unet_encoder_config)

    def forward(self, x, timesteps, context):
        out = self.unet_encoder(x, timesteps, context=context)
        return out


class UNetDecoderWrapper(pl.LightningModule):
    def __init__(self, unet_decoder_config):
        super().__init__()
        self.unet_decoder = instantiate_from_config(unet_decoder_config)

    def forward(self, h, emb, tp, hs, context) -> Any:
        out = self.unet_decoder(h, emb, tp, hs, context=context)
        return out
