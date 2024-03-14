import math

import numpy as np
import torch
from tqdm import tqdm

from modules.utils import get_device


def cosine_beta_schedule(
    timesteps: int, beta_start: float = 0.0, beta_end: float = 0.999, s=0.008
):
    """
    Cosine beta schedule for the DDIM model.
    :param timesteps: int: number of timesteps
    :param beta_start: float: start value of beta
    :param beta_end: float: end value of beta
    :param s: float: scaling factor
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
    ):
        if beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
            self.betas = torch.tensor(self.betas)
        elif beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_train_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = (
            np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_prev_t = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        variance = (beta_prod_prev_t / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_prev_t
        )
        return variance

    def set_timesteps(self, num_inference_steps: int, offset: int = 0):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, 1000, 1000 // num_inference_steps)[::-1].copy()
        self.timesteps += offset

    def step(
        self,
        model_output,
        timestep: int,
        sample,
        eta,
        use_clipped_model_output: bool,
        generator,
    ):
        # get previous timestep value
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # compute alphas and betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_prev_t = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # compute predicted original sample from predicted noise
        pred_original_sample = (
            sample - beta_prod_t**2 * model_output
        ) / alpha_prod_t**0.5

        # clip the predicted original sample
        if self.clip_sample:
            pred_original_sample = torch.clip(pred_original_sample, -1, 1)

        # compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_t = eta * variance**0.5

        if use_clipped_model_output:
            model_output = (
                sample - alpha_prod_t**0.5 * pred_original_sample
            ) / beta_prod_t**0.5

        # compute direction
        pred_sample_direction = (
            1 - alpha_prod_prev_t - std_t**2
        ) ** 0.5 * model_output

        # compute x_t without random noise
        prev_sample = (
            alpha_prod_prev_t**0.5 * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device if torch.is_tensor(model_output) else "cpu"
            noise = torch.randn(model_output.shape, generator=generator).to(device)
            variance = self._get_variance(timestep, prev_timestep) ** 0.5 * eta * noise

            if not torch.is_tensor(model_output):
                variance = variance.numpy()

            prev_sample = prev_sample + variance

        return prev_sample

    def add_noise(self, original_sample, noise, timesteps):
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = match_shape(sqrt_alpha_prod, original_sample)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = match_shape(
            sqrt_one_minus_alpha_prod, original_sample
        )

        noisy_samples = (
            sqrt_alpha_prod * original_sample + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    @torch.no_grad()
    def generate(
        self,
        model,
        batch_size: int = 1,
        generator=None,
        eta: float = 1.0,
        use_clipped_model_output: bool = True,
        num_inference_steps: int = 50,
        output_type: str = "pil",
        device: str | None = None,
    ):
        if device is None:
            device = get_device()

        image = torch.randn(
            (batch_size, model.in_channels, model.image_size, model.image_size),
            generator=generator,
        ).to(device)
        self.set_timesteps(num_inference_steps)
        for timestep in tqdm(self.timesteps):
            image = self.step(
                model(image),
                timestep,
                image,
                eta,
                use_clipped_model_output,
                generator,
            )

        return image

    def __len__(self):
        return self.num_train_timesteps
