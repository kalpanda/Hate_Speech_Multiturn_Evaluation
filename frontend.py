# frontend.py

import streamlit as st
import pandas as pd
import version2

st.set_page_config(page_title="Contextual Hate Detector", layout="wide")
st.title("🕵️ Context-Sensitive Hate Speech Detector")

# 1) API Key
persp_key = st.text_input(
    "Perspective API Key",
    type="password",
    help="Get yours from GCP Console"
)

# 2) Upload CSV
uploaded = st.file_uploader(
    "Upload chat CSV",
    type="csv",
    help="Columns: 'text', 'label'(0/1), plus 'previous_turn_i'"
)
# 3) Max Context Slider
max_ctx = st.slider(
    "Max context turns",
    0, 10, 1,
    help="We’ll test contexts from 0 up to this value"
)
info_order=st.text_input("The info order of the message to check")

if st.button("Run Detection"):
    if not uploaded:
        st.error("Please upload your CSV.")
    elif not persp_key.strip():
        st.error("Please enter your Perspective API key.")
    else:
        # Read CSV
        df = pd.read_csv(uploaded)
        # Run detectors
        with st.spinner("Processing…"):
            results = version2.run_all(df,info_order, persp_key,max_ctx)

        st.success("Done!")
        st.subheader("Details:")
        st.text(results)

