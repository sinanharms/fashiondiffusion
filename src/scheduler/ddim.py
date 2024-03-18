from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from modules.utils import get_device
from scheduler.util import make_ddim_schedule


class DDIMSampler(object):
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
            ddim_num_steps=ddim_num_steps,
            num_ddpm_steps=self.ddpm_num_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )

        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for all timesteps"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)
