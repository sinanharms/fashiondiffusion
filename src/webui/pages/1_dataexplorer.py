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
unique_products = np.unique(dataset.file["input_name"][:])


def configure_sidebar():
    with st.sidebar:
        with st.expander("Explorer Options"):
            st.write("Select the product to explore")
            st.write("Select the number of samples to display")
            random_samples = st.checkbox("Random samples", value=False)
            num_samples = st.slider(
                "Number of samples to display", min_value=1, max_value=100, value=10
            )
            if not random_samples:
                selected_product = st.selectbox(
                    label="Select Product", options=unique_products.tolist()
                )

    return random_samples, num_samples, selected_product


def main_page(selected_product, random_samples, num_samples: int):
    if not random_samples:
        idx = np.where(dataset.file["input_name"][:] == selected_product)[0]
        samples = [dataset.__get_sample__(i) for i in idx]
        for i in range(len(samples)):
            st.write(f"Sample {i+1}/{len(samples)}")
            st.image(
                samples[i]["image"],
                caption=samples[i]["description"],
                use_column_width=True,
            )
    else:
        idx = random.sample(range(dataset.length), num_samples)
        samples = [dataset.__get_sample__(i) for i in idx]
        for i in range(len(samples)):
            st.write(f"Sample {i+1}/{num_samples}")
            st.image(
                samples[i]["image"],
                caption=samples[i]["description"],
                use_column_width=True,
            )


def main():
    selected_product, random_samples, num_samples = configure_sidebar()
    main_page(selected_product, random_samples, num_samples)


if __name__ == "__main__":
    main()
