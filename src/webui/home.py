import streamlit as st

st.set_page_config(
    page_title="FashionGen: Generate Fashion Images",
    page_icon=":shirt:",
    layout="wide",
    initial_sidebar_state="expanded",
)
if "shared" is not st.session_state:
    st.session_state["shared"] = True


st.write("# FashionGen: Generate Fashion Images")
st.sidebar.success("Welcome to FashionGen!")

st.markdown(
    """
    FashionGen is a generative model that can generate fashion images.
    """
)
