from contextlib import contextmanager
from functools import partial
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from typing_extensions import Self

from modules.utils import count_params, default
from scheduler.util import extract_into_tensor, noise_like

__conditioning_keys__ = {
    "concat": "c_concat",
    "crossattn": "c_crossattn",
}


class Diffusion(pl.LightningModule):
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init: float = 0.0,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], f"Invalid parameterization {parameterization}"
        self.parameterization = parameterization
        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_stage_model = None
        self.clip_denosied = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema

        if self.use_ema:
            self.model_ema = EMA(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, load_only_unet=load_only_unet
            )

        self.register_schedule(
            betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(
        self,
        betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if betas is not None:
            betas = betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alpha_cumprod = np.cumprod(alphas, axis=0)
        alpha_cumprod_prev = np.append(1.0, alpha_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alpha_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for all timesteps"

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alpha_cumprod", to_torch(alpha_cumprod))
        self.register_buffer("alpha_cumprod_prev", to_torch(alpha_cumprod_prev))

        self.register_buffer("sqrt_alpha_cumprod", to_torch(np.sqrt(alpha_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", to_torch(np.sqrt(1.0 - alpha_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alpha_cumprod", to_torch(np.log(1.0 - alpha_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alpha_cumprod", to_torch(np.sqrt(1.0 / alpha_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alpha_cumprod", to_torch(np.sqrt(1.0 / alpha_cumprod - 1))
        )

        posterior_variance = (1 - self.v_posterior) * betas * (
            1 - alpha_cumprod_prev
        ) / (1 - alpha_cumprod) + self.v_posterior * betas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coeff1",
            to_torch(betas * np.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alpha_cumprod_prev) * np.sqrt(alphas) / (1.0 - alpha_cumprod)
            ),
        )
        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_log_variance
                * to_torch(alphas)
                * (1 - self.alpha_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alpha_cumprod))
                / (2.0 * 1 - torch.Tensor(alpha_cumprod))
            )
        else:
            raise NotImplementedError("Invalid parameterization")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: switched back to original weights")

    def init_from_ckpt(self, path, ignore_keys: list, only_model=False):
        ckpt = torch.load(path, map_location="cpu")
        if "state_dict" in list(ckpt.keys()):
            ckpt = ckpt["state_dict"]
        keys = list(ckpt.keys())
        for key in keys:
            for ik in ignore_keys:
                if key.startswith(ik):
                    print(f"Ignoring key {key}")
                    del ckpt[key]

        missing, unexpected = (
            self.load_state_dict(ckpt, strict=False)
            if not only_model
            else self.model.load_state_dict(ckpt, strict=False)
        )
        print(
            f"Restored from {ckpt} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def q_variance(self, x_start, t):
        """
        Compute the variance of the posterior distribution at time t
        """
        mean = extract_into_tensor(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alpha_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alpha_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict the starting point from noise
        """
        return (
            extract_into_tensor(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alpha_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the posterior distribution at time t
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coeff1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(
            self.posterior_variance, t, x_start.shape
        )
        posterior_log_variance = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_start.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x, t, clip_denoised: bool):
        """
        Compute the mean and variance of the prior distribution at time t
        """
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), t, device=device, dtype=torch.long),
                clip_denoised=self.clip_denosied,
            )
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape)
            * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (pred - target).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = F.mse_loss(target, pred)
            else:
                loss = F.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")
        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start, t, noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError("Invalid parameterization")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=(1, 2, 3))
        log_prefix = "train/" if self.training else "val/"
        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight
        loss_lvlb = (loss * self.lvlb_weights[t]).mean()
        loss = loss_simple + self.original_elbo_weight * loss_lvlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = x.rearrange(x, "b h c w -> b c h w")
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {
                key + "_ema": loss_dict_no_ema[key] for key in loss_dict_ema
            }
        self.log_dict(
            loss_dict_no_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_img_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_img_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_rows=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_rows = min(x.shape[0], n_rows)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        diffusion_row = []
        x_start = x[:n_rows]
        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor(t), "1 -> b", b=n_rows)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start, t, noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            with self.ema_scope("Plotting"):
                sample, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )
            log["samples"] = sample
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self) -> OptimizerLRScheduler:
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, unet_config, conditioning_key=None):
        super().__init__()
        self.diffusion_model = instantiate_from_config(unet_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, dim=1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "both":
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, dim=1)
            out = self.diffusion_model(xc, t, context=cc)
        else:
            raise NotImplementedError(
                f"Invalid conditioning key {self.conditioning_key}"
            )
        return out