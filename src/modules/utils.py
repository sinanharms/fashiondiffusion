import importlib
import math
import os
from inspect import isfunction
from typing import Optional, Union

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
from einops import repeat
from loguru import logger
from PIL import Image


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def checkpoint(func, inputs, param, flag):
    if flag:
        args = tuple(inputs) + tuple(param)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output = ctx.run_function(*ctx.input_tensors)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output = ctx.run_function(*shallow_copies)
        grad_input = torch.autograd.grad(
            output, ctx.input_tensors + ctx.input_params, grad_output, allow_unused=True
        )
        del ctx.input_tensors
        del ctx.input_params
        del output
        return (None, None) + grad_input


def linear(*args, **kwargs):
    """
    A wrapper around torch.nn.Linear to allow for easy mocking in tests.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    A wrapper around torch.nn.AvgPool2d to allow for easy mocking in tests.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError("Invalid number of dimensions")


def conv_nd(dims, *args, **kwargs):
    """
    A wrapper around torch.nn.Conv2d to allow for easy mocking in tests.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError("Invalid number of dimensions")


def zero_module(module):
    """
    A wrapper around torch.nn.Module to allow for easy mocking in tests.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """
    A wrapper around torch.nn.BatchNorm2d to allow for easy mocking in tests.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only: bool = False):
    """
    create sinusoidal timestep embeddings

    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    else:
        emb = repeat(timesteps, "b -> b d", d=dim)
    return emb


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()
    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)
    return values


def unnormalize_to_zero_one(tensor):
    return (tensor + 1) / 2


def numpy_to_pil(array):
    if array.ndim == 3:
        array = array[None, ...]

    image = (array * 255).round().astype("uint8")
    if image.shape[-1] == 1:
        pil_images = [Image.fromarray(img.squeeze(), mode="L") for img in image]
    else:
        pil_images = [Image.fromarray(img) for img in image]

    return pil_images


def count_params(model, verbose=False):
    """
    Count the number of parameters in a .model.
    """
    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {num_params * 1.e-6:.2f} M params.")
    return num_params


def default(value, d):
    if value is not None:
        return value
    return d() if isfunction(d) else d


def instantiate_from_config(config):
    """
    Instantiate a class from a configuration dictionary.
    """
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Config must have a target key.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def load_model_from_config(config, device: str, ckpt_path=None, verbose=False):
    """
    Load a model from a configuration dictionary.
    """
    logger.info(f"Instantiating .model from config: {config}")
    logger.info(f"Loading model from checkpoint: {ckpt_path}")
    pl_sd = load_state_dict(ckpt_path, device=device)
    if "global_step" in pl_sd:
        logger.info(f"Global Step: {pl_sd['global_step']}")
    state_dict = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(state_dict, strict=False)
    if len(m) > 0 and verbose:
        logger.info(f"Missing keys: {m}")
        print(m)
    if len(u) > 0 and verbose:
        logger.info(f"Unexpected keys: {u}")
        print(u)

    model.to(device)
    model.eval()
    return model


def get_obj_from_str(obj_str, reload=False):
    """
    Get an object from a string.
    """
    module, cls = obj_str.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module), cls)


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


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    device: str = "cpu",
    variant: Optional[str] = None,
):
    """
    Load a state dict from a checkpoint file.

    Args:
        checkpoint_file: The path to the checkpoint file.
        variant: The variant of the checkpoint file.

    Returns:
        The state dict.
    """
    try:
        weights_only_kwarg = {"weights_only": True}
        if device == torch.device("mps"):
            state_dict = torch.load(
                checkpoint_file, map_location="cpu", **weights_only_kwarg
            )
            state_dict = convert_state_dict_to_float32(state_dict)
        else:
            state_dict = torch.load(
                checkpoint_file, map_location=device, **weights_only_kwarg
            )
        return state_dict
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
            )


def convert_state_dict_to_float32(state_dict):
    """
    Convert the state dict to float32.
    """
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            state_dict[k] = v.float()
    return state_dict
