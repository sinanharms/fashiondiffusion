import numpy as np
import torch


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


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule, n_timesteps, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timesteps, dtype=torch.float64
            )
            ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timesteps + 1, dtype=torch.float64) / n_timesteps + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.99)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timesteps, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timesteps, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas.numpy()
