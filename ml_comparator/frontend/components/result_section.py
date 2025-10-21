from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
import streamlit as st


def _build_metrics_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for model in results.get("models", []):
        metrics = model.get("metrics", {})
        record = {
            "Model": model.get("name"),
            "Accuracy": metrics.get("accuracy"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1 Score": metrics.get("f1_score"),
            "ROC AUC": metrics.get("roc_auc"),
            "Training Time (s)": round(model.get("training_time", 0.0), 3),
        }
        records.append(record)
    return pd.DataFrame(records)


def render_results(api_url: str) -> None:
    """Render the model comparison results."""
    st.header("3. Review Results")

    results: Dict[str, Any] = st.session_state.get("training_results")
    if not results:
        st.info("Train at least one model to view the comparison report.")
        return

    best_model = results.get("best_model")
    class_labels = results.get("class_labels", [])

    metrics_df = _build_metrics_dataframe(results)
    if metrics_df.empty:
        st.warning("No metrics returned from the backend.")
        return

    st.subheader("Model performance overview")
    styled_df = metrics_df.set_index("Model").style.format("{:.4f}", na_rep="N/A").highlight_max(
        axis=0, color="#D6F5D6"
    )
    st.dataframe(styled_df, use_container_width=True)

    st.markdown(f"**Best model:** `{best_model}`")

    chart_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )
    bar_fig = px.bar(
        chart_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".2f",
        title="Accuracy and F1-score comparison",
    )
    bar_fig.update_layout(legend_title_text="Metric")
    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Confusion matrices")
    for model in results.get("models", []):
        matrix = np.array(model.get("confusion_matrix", []))
        if matrix.size == 0:
            continue
        heatmap = ff.create_annotated_heatmap(
            z=matrix,
            x=class_labels,
            y=class_labels,
            colorscale="Blues",
            showscale=True,
        )
        heatmap.update_layout(
            title=f"Confusion matrix: {model.get('name')}",
            xaxis_title="Predicted label",
            yaxis_title="True label",
        )
        st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Export report")
    col_csv, col_pdf = st.columns(2)
    with col_csv:
        try:
            csv_response = requests.get(f"{api_url}/download", params={"format": "csv"}, timeout=120)
            csv_response.raise_for_status()
            st.download_button(
                label="Download CSV",
                data=csv_response.content,
                file_name="model_comparison.csv",
                mime="text/csv",
            )
        except requests.RequestException as exc:
            st.error(f"Unable to generate CSV report: {exc}")

    with col_pdf:
        try:
            pdf_response = requests.get(f"{api_url}/download", params={"format": "pdf"}, timeout=120)
            pdf_response.raise_for_status()
            st.download_button(
                label="Download PDF",
                data=pdf_response.content,
                file_name="model_comparison.pdf",
                mime="application/pdf",
            )
        except requests.RequestException as exc:
            st.error(f"Unable to generate PDF report: {exc}")


