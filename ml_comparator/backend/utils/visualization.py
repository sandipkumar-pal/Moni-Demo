from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def metrics_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert metrics dictionary to a pandas DataFrame."""
    frame = pd.DataFrame.from_dict(results, orient="index")
    frame.index.name = "model"
    return frame.reset_index()


def export_metrics_csv(results: Dict[str, Dict[str, float]], output_path: Path) -> Path:
    """Save metrics to CSV file."""
    df = metrics_to_dataframe(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def export_metrics_pdf(results: Dict[str, Dict[str, float]], output_path: Path) -> Path:
    """Create a simple PDF report with model metrics."""
    df = metrics_to_dataframe(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.5), max(3, len(df) * 0.6)))
    ax.axis("off")
    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path


def get_best_model_name(metrics: Dict[str, Dict[str, float]], primary_metric: str = "f1_score") -> str:
    """Determine the model with the best performance for a specific metric."""
    if not metrics:
        return ""
    sorted_models: List[str] = sorted(
        metrics,
        key=lambda model: (
            metrics[model].get(primary_metric, float("nan")),
            metrics[model].get("accuracy", float("nan")),
        ),
        reverse=True,
    )
    return sorted_models[0]


