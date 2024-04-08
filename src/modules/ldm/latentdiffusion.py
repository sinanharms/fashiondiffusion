from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm

from src.modules.diffusionmodules.diagonalgaussian import DiagonalGaussian
from src.modules.encoders.vae import SimpleVAE
from src.modules.ldm.diffusion import Diffusion, __conditioning_keys__
from src.modules.utils import default, instantiate_from_config

__cond_stage_keys__ = ["image", "LR_image", "segmentation", "bbox_img"]

from src.scheduler.ddim import DDIMSampler
from src.scheduler.util import extract_into_tensor, noise_like


def disable_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentDiffusion(Diffusion):
    def __init__(
        self,
        first_stage_config,
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
        assert self.num_timesteps_cond <= kwargs["num_timesteps"]
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            cond_stage_config = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denosied = False
        self.bbox_tokenizer = None

        self.restart_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.restart_from_ckpt = True

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx):
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and self.restart_from_ckpt
        ):
            assert (
                self.scale_factor == 1.0
            ), "don't use custom rescaling and std-rescaling simultaneously"
            print("Rescaling by std")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"New scale factor: {self.scale_factor}")

    def register_schedule(
        self,
        betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            betas=betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disable_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disable_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(
        self, samples: list, desc="Denoising", force_no_denoiser_quantization=False
    ):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoised = self.decode_first_stage(
                zd.to(self.device, force_not_quantized=force_no_denoiser_quantization)
            )
            denoise_row.append(denoised)

        n_images_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row, dim=0)
        denoise_grid = rearrange(denoise_row, " n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_images_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussian):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError
        return z * self.scale_factor

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussian):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, height, width):
        x_t = torch.arange(0, width).view(1, width, 1).repeat(height, 1, 1)
        y_t = torch.arange(0, height).view(height, 1, 1).repeat(1, width, 1)
        arr = torch.cat([x_t, y_t], dim=-1)
        return arr

    def delta_border(self, height, width):
        lower_right_corner = torch.tensor([height - 1, width - 1]).view(1, 1, 2)
        mesh = self.meshgrid(height, width) / lower_right_corner
        dist_left_up = torch.min(mesh, dim=-1, keepdim=True)[0]
        dist_right_down = torch.min(1 - mesh, dim=-1, keepdim=True)[0]
        dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return dist

    def get_weighting(self, height, width, Ly, Lx, device):
        weighting = self.delta_border(height, width)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = (
            weighting.view(1, height * width, 1).repeat(1, 1, Ly * Lx).to(device)
        )
        if self.split_input_params["tie_breaker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting,
                self.split_input_params["clip_min_tie_weight"],
                self.split_input_params["clip_max_tie_weight"],
            )
            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        bs, nc, h, w = x.shape

        # number of crops per image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        unfold_params = dict(
            kernel_size=kernel_size, dilation=1, padding=1, stride=stride
        )
        unfold = torch.nn.Unfold(**unfold_params)
        if uf == 1 and df == 1:
            fold = torch.nn.Fold(output_size=x[2:], **unfold_params)
            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))
        elif uf > 1 and df == 1:
            fold_params = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[1] * uf),
                dilation=1,
                padding=1,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params
            )
            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )
        elif uf == 1 and df > 1:
            fold_params = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[1] // df),
                dilation=1,
                padding=0,
                stride=stride,
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params
            )
            weighting = self.get_weighting(
                kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)
            weighting = weighting.view(
                (1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx)
            )
        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ["caption", "coordinates_bbox"]:
                    c = batch[cond_key]
                elif cond_key == "class_label":
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c_key = __conditioning_keys__[self.model.conditioning_key]
                c = {c_key: c, "pos_x": pos_x, "pos_y": pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, force_not_quantized=False):
        z = 1.0 / self.scale_factor * z
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    logger.warning(
                        "Warning: kernel size is larger than image size. \n Reducing kernel..."
                    )
                    ks = (min(ks[0], h), min(ks[1], w))
                if stride[0] > h or stride[1] > w:
                    logger.warning(
                        "Warning: stride is larger than image size. \n Reducing stride..."
                    )
                    stride = (min(stride[0], h), min(stride[1], w))

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )
                z = unfold(z)
                # 1. reshape to image size
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
                # 2. apply model loop over last dim
                # force quantized if vq model needed
                output_list = [
                    self.first_stage_model.decode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]
                o = torch.stack(output_list, axis=-1)
                o = o * weighting
                # reverse 1. reshape to image size
                o = o.view((o.shape[0], -1, o.shape[-1]))
                # stitch together
                decoded = fold(o) / normalization
                return decoded
            else:
                # force quantized if vq model needed
                return self.first_stage_model.decode(z)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    logger.warning(
                        "Warning: kernel size is larger than image size. \n Reducing kernel..."
                    )
                    ks = (min(ks[0], h), min(ks[1], w))
                if stride[0] > h or stride[1] > w:
                    logger.warning(
                        "Warning: stride is larger than image size. \n Reducing stride..."
                    )
                    stride = (min(stride[0], h), min(stride[1], w))
                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    x, ks, stride, df=df
                )
                z = unfold(x)
                # 1. reshape to image size
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
                # 2. apply model loop over last dim
                output_list = [
                    self.first_stage_model.encode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]
                o = torch.stack(output_list, axis=-1)
                o = o * weighting
                # reverse 1. reshape to image size
                o = o.view((o.shape[0], -1, o.shape[-1]))
                # stitch together
                encoded = fold(o) / normalization
                return encoded
            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        if self.model.conditioning_key is not None:
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x, c, t)

    def apply_model(self, x_noisy, t, cond):
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.mode.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1
            ks = self.split_input_params["ks"]
            stride = self.split_input_params["stride"]
            h, w = x_noisy.shape[-2:]
            fold, unfold, normalization, weighting = self.get_fold_unfold(
                x_noisy, ks, stride
            )
            z = unfold(x_noisy)
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]
            if (
                self.cond_stage_key in __cond_stage_keys__
                and self.model.conditioning_key
            ):
                c_key = next(iter(cond.keys()))
                c = next(iter(cond.values()))
                assert len(c) == 1
                c = c[0]
                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))
                cond_list = [{c_key: c[:, :, :, :, i] for i in range(c.shape[-1])}]
            else:
                cond_list = [cond for _ in range(z.shape[-1])]
            output_list = [
                self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])
            ]
            assert not isinstance(output_list[0], tuple)
            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            o = o.view((o.shape[0], -1, o.shape[-1]))
            x_recon = fold(o) / normalization
        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple):
            x_recon = x_recon[0]
        return x_recon

    def predict_eps_from_start(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alpha_cumprod, t, x_t.shape)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t, c)
        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if self.parameterization == "eps":
            x_recon = self.predict_eps_from_start(x, t, model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t_in
        )
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        repeat_noise=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device=device, repeat=repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = F.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoise=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_t=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        if start_t is not None:
            timesteps = min(timesteps, start_t)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive denoising",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if isinstance(temperature, float):
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denosied,
                quantize_denoised=quantize_denoise,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img * mask + img_orig * (1.0 - mask)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_t=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_t is not None:
            timesteps = min(timesteps, start_t)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive sampling",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denosied,
                quantize_denoised=quantize_denoised,
            )
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img * mask + img_orig * (1.0 - mask)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(
                S=ddim_steps,
                shape=shape,
                conditioning=cond,
                batch_size=batch_size,
                verbose=False,
                **kwargs,
            )
        else:
            samples, intermediates = self.sample(
                cond, batch_size=batch_size, return_intermediates=True, **kwargs
            )
        return samples, intermediates

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_rows=2,
        sample=True,
        ddim_steps=200,
        ddim_eta=1.0,
        return_keys=None,
        quantize_denoised=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )
        N = min(x.shape[0], N)
        n_rows = min(x.shape[0], n_rows)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["cond"] = xc

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_rows]
            for t in range(self.num_timesteps):
                if t % self.log_every_t or t == self.num_timesteps - 1:
                    t = repeat((torch.tensor([t])), "1 -> b", b=n_rows)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoised"] = denoise_grid
            if quantize_denoised and not isinstance(self.first_stage_model, SimpleVAE):
                with self.ema_scope("Plotting quantized"):
                    samples, z_denoise_row = self.sample_log(
                        cond=c,
                        batch_size=N,
                        ddim=use_ddim,
                        ddim_steps=ddim_steps,
                        eta=ddim_eta,
                        quantize_denoised=True,
                    )
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_quantized"] = x_samples

            if plot_progressive_rows:
                with self.ema_scope("Plotting progressive"):
                    img, progressives = self.progressive_denoising(
                        cond=c,
                        shape=(self.channels, self.image_size, self.image_size),
                        batch_size=N,
                    )
                prog_row = self._get_denoise_row_from_list(
                    progressives, desc="Progressive denoising"
                )
                log["progressive"] = prog_row

            if return_keys:
                if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                    return log
                else:
                    return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self) -> OptimizerLRScheduler:
        learning_rate = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            logger.info(f"{self.__class__.__name__}: Conditioning stage is trainable.")
            params += list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            logger.info("Optimizing logvar.")
            params.append(self.logvar)
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            logger.info(f"Using scheduler: LambdaLR")
            scheduler = [
                {
                    "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [optimizer], scheduler
        return optimizer

    @torch.no_grad()
    def to_rqb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.rand(3, x.shape[1], 1, 1).to(x)
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
