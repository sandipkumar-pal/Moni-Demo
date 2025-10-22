from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st


def render_upload(api_url: str) -> None:
    """Render the dataset upload widget and send the file to the backend API."""
    st.header("1. Upload Dataset")
    st.write("Upload a structured dataset in CSV or Excel format to get started.")

    with st.form("upload_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="dataset_file",
            help="The uploaded data remains on this machine and is never shared.",
        )
        submitted = st.form_submit_button("Upload & Analyze", type="primary")

    if submitted and uploaded_file is None:
        st.warning("Please select a dataset file before uploading.")
        return

    if submitted and uploaded_file is not None:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }
        with st.spinner("Uploading dataset..."):
            try:
                response = requests.post(f"{api_url}/upload", files=files, timeout=120)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Failed to upload dataset: {exc}")
                return

        payload: Dict[str, Any] = response.json()
        st.success(payload.get("message", "Dataset uploaded successfully."))

        st.session_state["columns"] = payload.get("columns", [])
        st.session_state["dataset_summary"] = payload.get("summary", [])
        st.session_state["rows"] = payload.get("rows", 0)
        st.session_state.pop("training_results", None)

    if st.session_state.get("dataset_summary"):
        with st.expander("Dataset overview", expanded=True):
            st.write(
                f"**Rows:** {st.session_state.get('rows', 0)} | **Columns:** {len(st.session_state.get('columns', []))}"
            )
            summary_df = pd.DataFrame(st.session_state["dataset_summary"])
            st.dataframe(summary_df, use_container_width=True)


