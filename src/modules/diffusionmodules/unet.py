from abc import abstractmethod
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from modules.diffusionmodules.attention import AttentionBlock
from modules.diffusionmodules.diagonalgaussian import DiagonalGaussian
from modules.diffusionmodules.spatialattention import SpatialTransformer
from modules.diffusionmodules.unet_modules import UNetDecode, UNetEncode
from modules.encoders.autoencoder import AutoEncoder
from modules.ldm.diffusion import Diffusion
from modules.sampler.ddim import make_ddim_sampling_parameters, make_ddim_schedule
from modules.utils import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    default,
    extract_into_tensor,
    get_device,
    instantiate_from_config,
    linear,
    noise_like,
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


def disabled_train(self):
    """
    Overwrite model.train with this function to make sure train/eval mode
    does not change anymore.
    """
    return self


class UNetModel(nn.Module):
    """
    A UNet model for image embedding.
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


class UNet(Diffusion):
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

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        if (
            self.scale_factor == 1.0
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert (
                self.scale_factor == 1.0
            ), "Rather not use custom scale factor and std scaling together"
            # set rescale weight to 1./std of encoding
            logger.info("Using std-rescaling for UNet")
            x = super().get_input(batch)
            x = x.to(self._device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            logger.info(f"Setting rescale weight to {self.scale_factor}")

    def apply_model(self, x_noisy, timestep, conditioning):
        if not self.turbo:
            self.encode.to(self._device)

        step = self.unet_bs
        h, emb, hs = self.encode(x_noisy[0:step], timestep[:step], conditioning[:step])
        bs = conditioning.shape[0]

        len_hs = len(hs)

        for i in range(step, bs, step):
            h_temp, emb_temp, hs_temp = self.encode(
                x_noisy[i : i + step],
                timestep[i : i + step],
                conditioning[i : i + step],
            )
            h = torch.cat((h, h_temp), dim=0)
            emb = torch.cat((emb, emb_temp), dim=0)
            for j in range(len_hs):
                hs[j] = torch.cat((hs[j], hs_temp[j]), dim=0)

        if not self.turbo:
            self.encode.to("cpu")
            self.decode.to(self._device)

        hs_temp = [hs[j][:step] for j in range(len_hs)]
        x_recon = self.decode(
            h[:step], emb[:step], x_noisy.dtype, hs_temp, conditioning[:step]
        )
        for i in range(step, bs, step):
            hs_temp = [hs[j][i : i + step] for j in range(len_hs)]
            x_recon_temp = self.decode(
                h[i : i + step],
                emb[i : i + step],
                x_noisy.dtype,
                hs_temp,
                conditioning[i : i + step],
            )
            x_recon = torch.cat((x_recon, x_recon_temp), dim=0)

        if not self.turbo:
            self.decode.to("cpu")

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        self.ddim_timesteps = make_ddim_schedule(
            ddim_discr_method=ddim_discretize,
            num_ddim_steps=ddim_num_steps,
            num_ddpm_steps=self.num_timesteps,
            verbose=verbose,
        )

        assert (
            self.alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for all timesteps"

        to_torch = lambda x: x.to(self._device)
        self.register_buffer("betas", to_torch(self.betas))
        self.register_buffer("alphas_cumprod", to_torch(self.alphas_cumprod))
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alpha_prods=self.alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alpha", np.sqrt(1 - ddim_alphas))

    def sample(
        self,
        S,
        conditioning,
        x0=None,
        shape=None,
        seed=1234,
        callback=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        sampler="ddim",
        temperature=1.0,
        noise_dropout=0.0,
        score_corector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **kwargs,
    ):
        if self.turbo:
            self.encode.to(self._device)
            self.decode.to(self._device)

        if x0 is None:
            batch_size, b1, b2, b3 = shape
            img_shape = (1, b1, b2, b3)
            tens = []
            logger.info(f"seeds used = {[seed+s for s in range(batch_size)]}")
            for _ in range(batch_size):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=self._device))
                seed += 1
            noise = torch.cat(tens, dim=0)
            del tens

        x_latent = noise if x0 is None else x0

        # sample
        if sampler == "ddim":
            self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
            logger.info("Using DDIM sampling")
            samples = self.ddim_sampling(
                x_latent,
                conditioning,
                S,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                init_latent=x_T,
                use_original_steps=False,
            )
        elif sampler == "plms":
            self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
            logger.info("Using PLMs sampling")
            logger.info(f"Data shape for PLMs: {shape}")
            samples = self.plms_sampling(
                conditioning,
                batch_size,
                x_latent,
                callback=callback,
                img_callback=img_callback,
                quantize_denoised=quantize_x0,
                mask=mask,
                x0=x0,
                ddim_use_original_steps=False,
                noise_dropout=noise_dropout,
                temperature=temperature,
                score_corector=score_corector,
                corrector_kwargs=corrector_kwargs,
                log_every_t=log_every_t,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        else:
            raise ValueError(f"Sampler {sampler} not supported")

        if self.turbo:
            self.encode.to("cpu")
            self.decode.to("cpu")

        return samples

    @torch.no_grad()
    def stochastic_encode(
        self, x0, t, seed, ddim_eta, ddim_steps, use_original_step=False, noise=None
    ):
        self.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        if noise is None:
            batch_size, b1, b2, b3 = x0.shape
            img_shape = (1, b1, b2, b3)
            tens = []
            logger.info(f"seeds used = {[seed + s for s in range(batch_size)]}")
            for _ in range(batch_size):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=self._device))
                seed += 1
            noise = torch.cat(tens, dim=0)
            del tens
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(self.ddim_sqrt_one_minus_alpha, t, x0.shape) * noise
        )

    @torch.no_grad()
    def add_noise(self, x0, t):
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        noise = torch.randn(x0.shape, device=x0.device)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise
        )

    @torch.no_grad()
    def ddim_sampling(
        self,
        x_latent,
        conditioning,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        mask=None,
        init_latent=None,
        use_original_steps=False,
    ):
        timesteps = self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        logger.info(f"Running DDIM Sampling with total steps: {total_steps}")

        iterator = tqdm(time_range, desc="Decoding Image", total=total_steps)
        x_dec = x_latent
        x0 = init_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, dtype=torch.long, device=x_latent.device
            )

            if mask is not None:
                x0_noisy = x0
                x_dec = x0_noisy * mask + x_dec * (1 - mask)

            x_dec = self.p_sample_ddim(
                x_dec,
                conditioning,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        if mask is not None:
            return x0 * mask + (1.0 - mask) * x_dec
        return x_dec

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    @torch.no_grad()
    def plms_sampling(
        self,
        cond,
        b,
        img,
        ddim_use_original_steps=False,
        callback=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        device = self.betas.device
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        logger.info(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="PLMS Sampler", total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full(
                (b,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_plms(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
            )
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

        return img

    @torch.no_grad()
    def p_sample_plms(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t = self.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.parameterization == "eps"
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )

            return e_t

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]
            ) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t


class UNetEncoderWrapper(pl.LightningModule):
    def __init__(self, unet_encoder_config):
        super().__init__()
        self.unet_encoder: UNetEncode = instantiate_from_config(unet_encoder_config)

    def forward(self, x, timesteps, context):
        out = self.unet_encoder(x, timesteps, context=context)
        return out


class UNetDecoderWrapper(pl.LightningModule):
    def __init__(self, unet_decoder_config):
        super().__init__()
        self.unet_decoder: UNetDecode = instantiate_from_config(unet_decoder_config)

    def forward(self, h, emb, tp, hs, context) -> Any:
        out = self.unet_decoder(h, emb, tp, hs, context=context)
        return out


class FirstStageModel(Diffusion):
    def __init__(
        self,
        first_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.channel_mult)
        except:
            self.num_downs = 0
        self.instantiate_first_stage(first_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.restarted_from_ckpt = True

    def instantiate_first_stage(self, first_stage_config):
        model = instantiate_from_config(first_stage_config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussian):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise ValueError("Invalid encoder posterior")
        return z * self.scale_factor

    @torch.no_grad()
    def decode_first_stage(self, z, force_not_quantized=False):
        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if isinstance(self.first_stage_model, AutoEncoder):
                self.first_stage_model.use_tiling = True
                return self.first_stage_model.decode(z)
            else:
                return self.first_stage_model.decode(z)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if isinstance(self.first_stage_model, AutoEncoder):
                self.first_stage_model.use_tiling = True
                return self.first_stage_model.encode(x)
            else:
                self.first_stage_model.use_tiling = False
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)


class CondStageModel(Diffusion):
    def __init__(
        self,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.restarted_from_ckpt = True

    def initiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                logger.info("Using first stage as cond stage")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                logger.info(
                    f"Training {self.__class__.__name__} as unconditional model"
                )
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_unconditional__"
            assert config != "__is_first_stage__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_model is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussian):
                    c = c.mode()
            else:
                self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
