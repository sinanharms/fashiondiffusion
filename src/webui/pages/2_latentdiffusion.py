import io
import time
import zipfile

import numpy as np
import requests
import streamlit as st
import torch
from einops import rearrange
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid
from tqdm import trange

from modules.ldm.latentdiffusion import LatentDiffusion
from modules.sampler.ddim import DDIMSampler
from modules.sampler.plms import PLMSSampler
from modules.utils import get_device, load_model_from_config

LATENT_CHANNELS: int = 4

st.sidebar.success("Latent Diffusion")

device = get_device()
logger.info(f"Using device: {device}")

config = OmegaConf.load("src/models/config.yaml")
model: LatentDiffusion = load_model_from_config(config)

model.to(device)


def configure_sidebar():
    """
    Configure the sidebar.
    """
    with st.sidebar:
        st.info("Latent Diffusion")
        with st.form("configure_form"):
            with st.expander("Configure the model"):
                width = st.number_input("Width", value=512)
                height = st.number_input("Height", value=512)
                num_outputs = st.slider(
                    "Number of outputs", min_value=1, max_value=10, value=1
                )
                sampling = st.selectbox("Scheduler", options=["DDIM", "PLMS"])
                num_iter = st.slider(
                    "Number of iterations", min_value=1, max_value=10, value=2
                )
                num_inference_steps = st.slider(
                    "Number of inference steps", min_value=1, max_value=1000, value=50
                )
                ddim_eta = st.slider(
                    "Eta for DDIM", min_value=0.0, max_value=1.0, value=0.0
                )
                guidance_scale = st.slider(
                    "Scale for classifier free guidance",
                    min_value=1.0,
                    max_value=50.0,
                    value=7.5,
                )
            prompt = st.text_area(
                "**Enter text prompt for image generation**",
                value="A dress with flowers",
            )

            submitted = st.form_submit_button(
                "Generate Image", type="primary", use_container_width=True
            )

        st.divider()
        st.markdown("Pre-trained Diffusion Model for Image Generation.")

    return (
        submitted,
        width,
        height,
        num_outputs,
        sampling,
        num_inference_steps,
        num_iter,
        ddim_eta,
        guidance_scale,
        prompt,
    )


def main_page(
    submitted: bool,
    width: int,
    height: int,
    num_outputs: int,
    sampling: str,
    num_inference_steps: int,
    num_iter: int,
    ddim_eta: float,
    guidance_scale: float,
    prompt: str,
):
    start_code = None
    if submitted:
        with st.status("Generating images...", expanded=True) as status:
            st.write("Model initialized.")
            try:
                if sampling == "DDIM":
                    sampler: DDIMSampler = DDIMSampler(model)
                elif sampling == "PLMS":
                    sampler: PLMSSampler = PLMSSampler(model)
                else:
                    raise NotImplementedError
                with torch.no_grad():
                    with model.ema_scope():
                        tic = time.time()
                        all_samples = list()
                        for _ in trange(num_iter, desc="Generating samples"):
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
                                verbose=False,
                                unconditional_guidance_scale=guidance_scale,
                                unconditional_conditioning=uc,
                                eta=ddim_eta,
                                start_code=start_code,
                            )
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                            )
                            x_samples_ddim = (
                                x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            )
                            x_sample_image_torch = torch.from_numpy(
                                x_samples_ddim
                            ).permute(0, 3, 1, 2)

                            for x_sample in x_sample_image_torch:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                img = Image.fromarray(x_sample.astype(np.uint8))

                            all_samples.append(x_samples_ddim)

                        toc = time.time()
                        status.success(f"Image generated in {toc - tic:.2f} seconds.")
                all_images = []
                if img:
                    st.session_state.generated_image = img
                    for image in st.session_state.generated_image:
                        with st.container():
                            st.image(image, use_column_width=True, caption=prompt)
                            all_images.append(image)

                            response = requests.get(image)
                st.session_state.all_images = all_images

                zip_io = io.BytesIO()

                with zipfile.ZipFile(zip_io, mode="w") as z:
                    for i, image in enumerate(st.session_state.all_images):
                        response = requests.get(image)
                        if response.status_code == 200:
                            z.writestr(f"image_{i}.png", response.content)
                        else:
                            st.error("Error downloading image.")
            except Exception as e:
                logger.info(e)
                status.error("Error generating image.")
    else:
        pass


def main():
    (
        submitted,
        width,
        height,
        num_outputs,
        sampling,
        num_inference_steps,
        num_iter,
        ddim_eta,
        guidance_scale,
        prompt,
    ) = configure_sidebar()
    main_page(
        submitted,
        width,
        height,
        num_outputs,
        sampling,
        num_inference_steps,
        num_iter,
        guidance_scale,
        prompt,
    )


if __name__ == "__main__":
    main()
