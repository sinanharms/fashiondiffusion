import math

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
        return CheckPointFunction.apply(func, len(inputs), *args)
    else:
        func(*inputs)


class CheckPointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output = ctx.run_function(*ctx.input_tensors)
        return output

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output
        return (None, None) + input_grads


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
    if not repeat_only:
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([args.cos(args), args.sin(args)], dim=-1)
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
