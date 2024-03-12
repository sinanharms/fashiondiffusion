from pathlib import Path

import streamlit as st

from data.fashiongen import FashionGenDataset

datapath = (
    Path(__file__).parent.parent.parent / "fashiongen" / "fashiongen_256_256_train.h5"
)
dataset = FashionGenDataset(datapath)


def data_explorer():
    st.title("FashionGen Data Explorer")
    st.write("This is simple data explorer for the FashionGen Dataset.")

    num_samples = st.slider("Number of samples to display", 1, 100, 10)
    samples = [dataset[i] for i in range(num_samples)]

    for i, (image, description) in enumerate(samples):
        st.image(image, caption=description, use_column_width=True)
        st.write(f"Sample {i+1}/{num_samples}")
