import streamlit as st
from loguru import logger
from omegaconf import OmegaConf

# from src.modules.sampler.ddim import DDIMSampler
# from src.modules.sampler.plms import PLMSSampler
# from src.modules.utils import get_device, load_model_from_config

# st.sidebar.success("Latent Diffusion")
#
# device = get_device()
# logger.info(f"Using device: {device}")
#
# config = OmegaConf.load("configs/latent_diffusion.yaml")
# model = load_model_from_config(config)
#
# model.to(device)
#
# if sampler == "DDIM":
#     sampler = DDIMSampler(model)
# elif sampler == "PLMS":
#     sampler = PLMSSampler(model)
# else:
#     raise NotImplementedError


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
                sampler = st.selectbox("Scheduler", options=["DDIM", "PLMS"])
                num_inference_steps = st.slider(
                    "Number of inference steps", min_value=1, max_value=1000, value=50
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
        sampler,
        num_inference_steps,
        guidance_scale,
        prompt,
    )


if __name__ == "__main__":
    (
        submitted,
        width,
        height,
        num_outputs,
        sampler,
        num_inference_steps,
        guidance_scale,
        prompt,
    ) = configure_sidebar()
    if submitted:
        st.write("Generating images...")
        # image = sampler.sample(prompt, width, height, num_outputs, num_inference_steps, guidance_scale)
        # st.image(image, width=512)
        st.write("Images generated.")
