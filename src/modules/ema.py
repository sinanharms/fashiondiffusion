import torch
from loguru import logger
from torch import nn


class EMA(nn.Module):
    def __init__(self, model, decay=0.99, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int)
            if use_num_updates
            else torch.tensor(-1, dtype=torch.int),
        )
        self.parameter_names_dict = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                safe_name = name.replace(".", "_")
                self.parameter_names_dict.update({name: safe_name})
                self.register_buffer(safe_name, param.clone().detach().date)

        self.collected_params = []

    def forward(self, model):
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in model_params:
                if model_params[key].requires_grad:
                    safe_name = self.parameter_names_dict[key]
                    shadow_params[safe_name] = shadow_params[safe_name].type_as(
                        model_params[key]
                    )
                    shadow_params[safe_name].sub_(
                        one_minus_decay * (shadow_params[safe_name] - model_params[key])
                    )
                else:
                    assert (
                        not key in self.parameter_names_dict
                    ), f"key {key} is not in parameter_names_dict"

    def copy_to(self, model):
        model_params = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in model_params:
            if model_params[key].requires_grad:
                model_params[key].data.copy_(
                    shadow_params[self.parameter_names_dict[key]].data
                )
            else:
                assert (
                    not key in self.parameter_names_dict
                ), f"key {key} is not in parameter_names_dict"

    def restore(self, parameters):
        for collected_param, param in zip(self.collected_params, parameters):
            param.data.copy_(collected_param.data)
