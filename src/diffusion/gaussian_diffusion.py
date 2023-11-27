from pathlib import Path

import torch
from ldm.models.diffusion.ddim import DDimSampler
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, device, verbose=False):
    if isinstance(config, (str, Path)):
        pass

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model
