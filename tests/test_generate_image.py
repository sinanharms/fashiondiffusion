import time

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange

from modules.ldm.latentdiffusion import LatentDiffusion
from modules.sampler.ddim import DDIMSampler
from modules.utils import get_device, load_model_from_config

if __name__ == "__main__":
    config = OmegaConf.load("../src/config/txt2img.yaml")
    ckpt = "../src/.model/ldm/model.ckpt"
    device = get_device()

    LATENT_CHANNELS: int = 4
    num_iter = 2
    guidance_scale = 7.5
    num_outputs = 1
    num_inference_steps = 50
    ddim_eta = 0.0
    height = 512
    width = 512

    model: LatentDiffusion = load_model_from_config(config, ckpt)
    model.to(device)

    sampler = DDIMSampler(model)
    with torch.no_grad():
        with model.model_ema():
            tic = time.time()
            all_samples = list()
            for _ in trange(num_iter, desc="Generating samples..."):
                uc = None
                if guidance_scale != 1.0:
                    uc = model.get_learned_conditioning(num_outputs * [""])
                if isinstance(prompt, tuple):
                    prompt = list(prompt)
                c = model.get_learned_conditioning(prompt)
                shape = [LATENT_CHANNELS, height, width]
                samples_ddim, _ = sampler.sample(
                    S=num_inference_steps,
                    c=c,
                    batch_size=num_outputs,
                    shape=shape,
                    verbose=True,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_sample_image_torch = torch.from_numpy(x_samples_ddim).permute(
                    0, 3, 1, 2
                )
                for x_sample in x_sample_image_torch:
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    all_samples.append(img)
            toc = time.time()
