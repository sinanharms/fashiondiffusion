import streamlit as st

st.sidebar.success("Latent Diffusion")
st.sidebar.markdown("This page is under construction.")

prompt = st.sidebar.text_input(
    label="text prompt for image generation", value="A dress with flowers"
)
sampler = st.sidebar.selectbox(label="Sampler", options=["DDIM", "DDPM"])
