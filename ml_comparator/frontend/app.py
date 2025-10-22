from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


if __package__ is None or __package__ == "":  # pragma: no cover - runtime import guard
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from ml_comparator.frontend.components.result_section import render_results
from ml_comparator.frontend.components.selection_section import render_selection
from ml_comparator.frontend.components.upload_section import render_upload


st.set_page_config(page_title="ML Classification Comparator", layout="wide")
st.title("ML Classification Comparator")
st.write(
    "Upload a dataset, choose multiple machine learning classifiers, and compare their performance side by side."
)

if "api_url" not in st.session_state:
    st.session_state["api_url"] = "http://localhost:8000"

with st.sidebar:
    st.header("Backend settings")
    st.session_state["api_url"] = st.text_input(
        "FastAPI backend URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        help="The Streamlit app communicates with this FastAPI server.",
    )
    st.markdown(
        "Ensure the FastAPI server is running before training models. Use the provided Dockerfile or run the backend manually."
    )

api_url = st.session_state["api_url"].rstrip("/")

render_upload(api_url)
render_selection(api_url)
render_results(api_url)


