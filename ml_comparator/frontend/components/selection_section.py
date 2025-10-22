from __future__ import annotations

from typing import Dict, List

import requests
import streamlit as st

AVAILABLE_MODELS: List[str] = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "SVC",
    "KNeighborsClassifier",
    "GaussianNB",
    "XGBClassifier",
]


def render_selection(api_url: str) -> None:
    """Render algorithm selection and trigger model training."""
    st.header("2. Configure Experiment")

    if "columns" not in st.session_state or not st.session_state["columns"]:
        st.info("Upload a dataset to configure training.")
        return

    target_column = st.selectbox(
        "Select target column",
        options=st.session_state["columns"],
        key="target_column",
    )

    default_models = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]
    selected_models = st.multiselect(
        "Select classification algorithms",
        options=AVAILABLE_MODELS,
        default=[model for model in default_models if model in AVAILABLE_MODELS],
        help="Choose one or more models to compare.",
    )

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider(
            "Test size (fraction of data used for testing)",
            min_value=0.1,
            max_value=0.4,
            step=0.05,
            value=0.2,
        )
    with col2:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            max_value=10_000,
            value=42,
            step=1,
        )

    if st.button("Run Training", type="primary", use_container_width=True):
        payload: Dict[str, object] = {
            "target_column": target_column,
            "algorithms": selected_models,
            "test_size": test_size,
            "random_state": int(random_state),
        }

        if not selected_models:
            st.warning("Please select at least one algorithm before training.")
            return

        with st.spinner("Training models..."):
            try:
                response = requests.post(f"{api_url}/train", json=payload, timeout=600)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Training failed: {exc}")
                return

        st.session_state["training_results"] = response.json()
        st.success("Training completed successfully.")


