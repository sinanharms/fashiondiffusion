import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from modules.utils import extract_into_tensor, get_device, noise_like


def make_ddim_schedule(ddim_discr_method, num_ddim_steps, num_ddpm_steps, verbose=True):
    if ddim_discr_method == "uniform":
        c = num_ddpm_steps // num_ddim_steps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_steps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_steps * 0.8), num_ddim_steps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f"Discretization method {ddim_discr_method} not implemented"
        )

    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"DDIM timesteps: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(alpha_prods, ddim_timesteps, eta, verbose=True):
    alphas = alpha_prods[ddim_timesteps]
    alphas_prev = np.asarray(
        [alpha_prods[0]] + alpha_prods[ddim_timesteps[:-1].tolist()]
    )

    sigma = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f"alphas from ddim smaple: a_t: {alphas}, a_t-1: {alphas_prev}")
        print(f"sigma from ddim smaple: {sigma} for chosen values of eta: {eta}")
    return sigma, alphas, alphas_prev


class DDIMSampler(nn.Module):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.device = self.model.device if self.model.device else get_device()

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        self.ddim_timesteps = make_ddim_schedule(
            ddim_discr_method=ddim_discretize,
            num_ddim_steps=ddim_num_steps,
            num_ddpm_steps=self.ddpm_num_timesteps,
            verbose=verbose,
        )

        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for all timesteps"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and other parameters
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / (alphas_cumprod.cpu() - 1.0))),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alpha_prods=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", to_torch(ddim_sigmas))
        self.register_buffer("ddim_alphas", to_torch(ddim_alphas))
        self.register_buffer("ddim_alphas_prev", to_torch(ddim_alphas_prev))
        self.register_buffer(
            "ddim_sqrt_one_minus_alphas", to_torch(np.sqrt(1.0 - ddim_alphas))
        )
        sigmas_for_original_sampling_step = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "sigmas_for_original_sampling_step", sigmas_for_original_sampling_step
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
                        f"batch size {batch_size} does not match conditioning batch size {cbs}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    raise ValueError(
                        f"batch size {batch_size} does not match conditioning batch size {conditioning.shape[0]}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sample from the model
        C, H, W = shape
        size = (batch_size, C, H, W)
        logger.info(
            f"Data shape for DDIM sampling: {size} with batch size {batch_size} and eta {eta}"
        )
        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
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
    def ddim_sampling(
        self,
        conditioning,
        shape,
        callback=None,
        timesteps=None,
        img_callback=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        ddim_use_original_steps=False,
        noise_dropout=0.0,
        temperature=1.0,
        score_correction=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
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
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(
                min(timesteps / self.ddim_timesteps.shape[0], 1)
                * self.ddim_timesteps.shape[0]
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        iterator = tqdm(time_range, desc="DDIM Sampling", total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None, "mask requires x0 to be defined"
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_ddim(
                img,
                conditioning,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                noise_dropout=noise_dropout,
                temperature=temperature,
                score_correction=score_correction,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(i, img)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

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
        noise_dropout=0.0,
        temperature=1.0,
        score_correction=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        b, *_, device = *x.shape, x.device
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([c] * 2)
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        if score_correction is not None:
            assert self.model.parameterization == "eps"
            e_t = score_correction.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

        # select parameters to the currently sampled timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_t_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_a_t = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction of x0
        pred_x0 = (x - sqrt_one_minus_a_t * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_t_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(
            x.shape,
            device=device,
        )
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_t_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def denoise(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
    ):
        timesteps = (
            np.arange(self.ddpm_num_timesteps)
            if use_original_steps
            else self.ddim_timesteps
        )
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"DDIM Denoising with {total_steps} steps")

        iterator = tqdm(time_range, desc="DDIM Denoising Image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec
