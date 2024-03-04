import torch
import torch.nn as nn


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


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
