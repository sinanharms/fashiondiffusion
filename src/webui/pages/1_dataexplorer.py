import random
from pathlib import Path

import numpy as np
import streamlit as st

from data.fashiongen import FashionGenDataset

datapath = (
    Path(__file__).parent.parent.parent.parent
    / "fashiongen"
    / "fashiongen_256_256_train.h5"
)
dataset = FashionGenDataset(datapath)

st.sidebar.success("Dataexplorer")


st.title("FashionGen Data Explorer")


random_idx = st.sidebar.checkbox("Random samples", value=False)
if not random_idx:
    unique_products = np.unique(dataset.file["input_name"][:])
    selected_product = st.sidebar.selectbox(
        label="Select Product", options=unique_products.tolist()
    )
    idx = np.where(dataset.file["input_name"][:] == selected_product)[0]
    samples = [dataset.__get_sample__(i) for i in idx]
    for i in range(len(samples)):
        st.image(
            samples[i]["image"],
            caption=samples[i]["description"],
            use_column_width=True,
        )
        st.write(f"Sample {i+1}/{len(samples)}")
else:
    num_samples = st.sidebar.slider("Number of samples to display", 1, 100, 10)

    idx = range(num_samples)
    if random_idx:
        idx = random.sample(range(len(dataset)), num_samples)
    samples = [dataset.__get_sample__(i) for i in idx]

    for i in range(len(samples)):
        st.image(
            samples[i]["image"],
            caption=samples[i]["description"],
            use_column_width=True,
        )
        st.write(f"Sample {i+1}/{num_samples}")
