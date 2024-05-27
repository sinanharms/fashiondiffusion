import os
import re
import time
from contextlib import nullcontext
from itertools import islice
from random import randint

import numpy as np
import torch
from einops import rearrange
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from tqdm import tqdm, trange

from modules.utils import get_device, instantiate_from_config, load_model_from_config


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


if __name__ == "__main__":
    device = get_device()

    config = "../src/config/optimized_sd.yaml"
    DEFAULT_CKPT = "../src/.model/ldm/model.ckpt"
    outdir = "../.outputs/txt2img-samples"
    num_iter = 1
    guidance_scale = 7.5
    n_samples = 1
    num_inference_steps = 50
    ddim_eta = 0.0
    C: int = 4
    H = 16
    W = 16
    f = 1
    n_rows = 1
    fixed_code = False
    from_file = False
    prompt = "Red dress with sun as a flower print"
    precision = "autocast"
    sampler = "ddim"

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    grid_count = len(os.listdir(outpath)) - 1

    seed = randint(0, 1000000)
    seed_everything(seed)
    config = OmegaConf.load(f"{config}")
    sd = load_model_from_config(config, ckpt_path=DEFAULT_CKPT, device=device)
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    model._device = get_device()
    model.turbo = True

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = get_device()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if get_device() != "cpu":
        model.half()
        modelCS.half()

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        assert prompt is not None
        prompt = prompt
        print(f"Using prompt: {prompt}")
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            text = f.read()
            print(f"Using prompt: {text.strip()}")
            data = text.splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))

    if precision == "autocast" and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    with torch.no_grad():
        all_samples = list()
        for n in trange(num_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                sample_path = os.path.join(
                    outpath, "_".join(re.split(":| ", prompts[0]))
                )[:150]
                os.makedirs(sample_path, exist_ok=True)
                base_count = len(os.listdir(sample_path))

                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if guidance_scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    c = modelCS.get_learned_conditioning(prompts)

                    shape = [n_samples, C, H // f, W // f]

                    if device != "cpu" and device != "mps":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=num_inference_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=guidance_scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler=sampler,
                    )

                    modelFS.to(device)

                    print(samples_ddim.shape)
                    print("saving images")
                    for i in range(batch_size):
                        x_samples_ddim = modelFS.decode_first_stage(
                            samples_ddim[i].unsqueeze(0)
                        )
                        x_sample = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        x_sample = 255.0 * rearrange(
                            x_sample[0].cpu().numpy(), "c h w -> h w c"
                        )
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(
                                sample_path,
                                "seed_" + str(seed) + "_" + f"{base_count:05}.{format}",
                            )
                        )
                        seeds += str(seed) + ","
                        seed += 1
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0
