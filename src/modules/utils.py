import importlib
import math
from inspect import isfunction

import torch
import torch.nn as nn
from einops import repeat
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
    pil_images = [Image.fromarray(img) for img in image]

    return pil_images


def count_params(model, verbose=False):
    """
    Count the number of parameters in a model.
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
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Config must have a target key.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def get_obj_from_str(obj_str, reload=False):
    """
    Get an object from a string.
    """
    module, cls = obj_str.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module), cls)
