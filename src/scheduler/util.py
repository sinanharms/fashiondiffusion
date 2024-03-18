import numpy as np


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
    return alphas, sigma
