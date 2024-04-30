from copy import deepcopy
from functools import partial
from typing import Optional

import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn.modules.module import T


def exists(val):
    return val is not None


def inplace_copy(target: Tensor, source: Tensor, auto_move_device: bool = False):
    if auto_move_device:
        source = source.to(target.device)

    target.copy_(source)


def inplace_lerp(
    target: Tensor, source: Tensor, weight, *, auto_move_device: bool = False
):
    if auto_move_device:
        source = source.to(target.device)

    target.lerp_(source, weight)


class EMA(nn.Module):
    def __init__(
        self,
        model,
        ema_model: Optional[nn.Module] = None,
        beta=0.9999,
        update_after_step=100,
        update_freq=10,
        inv_gamma=1.0,
        power=0.33,
        min_value=0.0,
        allow_different_devices=False,
    ):
        super().__init__()
        self.beta = beta
        self.is_frozen = beta == 1.0
        self.ema_model = ema_model
        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                logger.error(f"Failed to deepcopy model: {e}")

        self.ema_model.requires_grad_(False)

        self.parameter_names = {
            name
            for name, param in self.ema_model.named_parameters()
            if torch.is_floating_point(param) or torch.is_complex(param)
        }
        self.buffer_names = {
            name
            for name, buffer in self.ema_model.named_buffers()
            if torch.is_floating_point(buffer) or torch.is_complex(buffer)
        }

        self.inplace_copy = partial(
            inplace_copy, auto_move_device=allow_different_devices
        )
        self.inplace_lerp = partial(
            inplace_lerp, auto_move_device=allow_different_devices
        )

        self.update_freq = update_freq
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.allow_different_devices = allow_different_devices

        # Init and step states
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0))

    def eval(self: T) -> T:
        return self.ema_model.eval()

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.model)
        ):
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(
            self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)
        ):
            copy(ma_buffers.data, current_buffers.data)

    def copy_params_form_ema_to_model(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.model)
        ):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(
            self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)
        ):
            copy(current_buffers.data, ma_buffers.data)

    def get_current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch.item() <= 0:
            return 0.0

        return value.clamp(min=self.min_value, max=self.beta).item()

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_freq) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))

        self.update_moving_average(self.ema_model, self.model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if self.is_frozen:
            return

        copy, lerp = self.inplace_copy, self.inplace_lerp
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            self.get_params_iter(current_model), self.get_params_iter(ma_model)
        ):
            lerp(ma_params.data, current_params.data, 1.0 - current_decay)

        for (name, current_buffers), (_, ma_buffers) in zip(
            self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)
        ):
            lerp(ma_buffers.data, current_buffers.data, 1.0 - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
