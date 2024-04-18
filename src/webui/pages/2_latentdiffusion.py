import streamlit as st
from loguru import logger

from modules.utils import get_device
from webui.util import generate

LATENT_CHANNELS: int = 4

st.sidebar.success("Latent Diffusion")

device = get_device()
logger.info(f"Using device: {device}")


with st.sidebar:
    st.info("Latent Diffusion")
    with st.form("configure_form"):
        with st.expander("Configure the .model"):
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


if submitted:
    with st.spinner("Generating samples..."):
        img = generate(
            prompt,
            width,
            height,
            num_outputs,
            sampling,
            num_iter,
            num_inference_steps,
            ddim_eta,
            guidance_scale,
        )
        st.image(img, caption=prompt, use_column_width=True)
