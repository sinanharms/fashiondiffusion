from functools import partial

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.modules.ldm.latentdiffusion import LatentDiffusion
from src.modules.sampler.ddim import make_ddim_sampling_parameters, make_ddim_schedule
from src.modules.utils import noise_like


class PLMSSampler:
    def __init__(self, model: LatentDiffusion, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        if ddim_eta != 0.0:
            raise ValueError("DDIM eta must be 0.")
        self.ddim_timesteps = make_ddim_schedule(
            ddim_discr_method=ddim_discretize,
            num_ddim_steps=ddim_num_steps,
            num_ddpm_steps=self.num_timesteps,
            verbose=verbose,
        )
        alpha_cumprod = self.model.alpha_cumprod
        assert alpha_cumprod.shape[0] == self.num_timesteps, "Invalid alphas_cumprod."
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alpha_cumprod", to_torch(alpha_cumprod))
        self.register_buffer(
            "alpha_cumprod_prev", to_torch(self.model.alpha_cumprod_prev)
        )

        self.register_buffer(
            "sqrt_alpha_cumprod", to_torch(np.sqrt(alpha_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            to_torch(np.sqrt(1.0 - alpha_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alpha_cumprod", to_torch(np.log(1.0 - alpha_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alpha_cumprod", to_torch(np.sqrt(1.0 / alpha_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alpha_cumprod",
            to_torch(np.sqrt(1.0 / alpha_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alpha_prods=alpha_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", to_torch(ddim_sigmas))
        self.register_buffer("ddim_alphas", to_torch(ddim_alphas))
        self.register_buffer("ddim_alphas_prev", to_torch(ddim_alphas_prev))
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alpha_cumprod_prev)
            / (1 - self.alpha_cumprod)
            * (1 - self.alpha_cumprod / self.alpha_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_correction=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    raise ValueError(
                        f"Conditioning batch size ({cbs}) must match sample batch size ({batch_size})."
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    raise ValueError(
                        f"Conditioning batch size ({conditioning.shape[0]}) must match sample batch size ({batch_size})."
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, H, W = shape
        size = (batch_size, C, H, W)
        logger.info(f"Sampling {batch_size} images of shape {size} and eta {eta}.")
        samples, intermediates = self.plms_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_x0=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_correction=score_correction,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling(
        self,
        conditioning,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
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
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(
                min(timesteps / self.ddim_timesteps.shape[0], 1)
                * self.ddim_timesteps.shape[0]
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x": [img]}
        time_range = (
            list(reversed(range(timesteps)))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        logger.info(f"PLMS Sampling with {total_steps} steps.")

        iterator = tqdm(time_range, desc="PLMS Sampling", total=total_steps)
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
                assert x0 is not None, "Mask requires x0."
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + x0 * (1.0 - mask)

            outs = self.p_sample_plms(
                img,
                conditioning,
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
                ts_next=ts_next,
            )
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x"].append(pred_x0)

        return img, intermediates

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
        ts_next=None,
    ):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert (
                    self.model.parameterization == "eps"
                ), "Score correction requires eps parameterization."
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )
            return e_t

        alphas = self.model.alpha_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alpha_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alpha_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

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
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
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
            e_t_next = get_model_output(x_prev, ts_next)
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
