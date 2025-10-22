from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .model_trainer import TrainingResponse, train_models
from .utils.visualization import (
    export_metrics_csv,
    export_metrics_pdf,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TEMP_DIR = DATA_DIR / "temp"

for directory in (UPLOAD_DIR, TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class TrainRequest(BaseModel):
    target_column: str
    algorithms: List[str]
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42


class DataStore:
    def __init__(self) -> None:
        self.file_path: Optional[Path] = None
        self.dataframe: Optional[pd.DataFrame] = None
        self.results: Optional[TrainingResponse] = None
        self.target_column: Optional[str] = None

    def set_dataset(self, file_path: Path, dataframe: pd.DataFrame) -> None:
        self.file_path = file_path
        self.dataframe = dataframe
        self.results = None
        self.target_column = None

    def set_results(self, target_column: str, results: TrainingResponse) -> None:
        self.target_column = target_column
        self.results = results

    def require_dataset(self) -> pd.DataFrame:
        if self.dataframe is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
        return self.dataframe

    def require_results(self) -> TrainingResponse:
        if self.results is None:
            raise HTTPException(status_code=400, detail="No training results available.")
        return self.results


def read_dataset(file_path: Path) -> pd.DataFrame:
    extension = file_path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(file_path)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")


def summarize_dataframe(df: pd.DataFrame) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        summary.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_count": int(series.count()),
                "null_count": int(series.isna().sum()),
            }
        )
    return summary


def training_response_to_dict(response: TrainingResponse) -> Dict[str, object]:
    return {
        "models": [
            {
                "name": model.name,
                "metrics": model.metrics,
                "confusion_matrix": model.confusion_matrix,
                "training_time": model.training_time,
            }
            for model in response.models
        ],
        "best_model": response.best_model,
        "metrics": response.metrics,
        "class_labels": response.class_labels,
        "feature_names": response.feature_names,
    }


data_store = DataStore()
app = FastAPI(title="ML Classification Comparator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    extension = Path(file.filename).suffix.lower()
    if extension not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    file_identifier = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{file_identifier}{extension}"

    content = await file.read()
    file_path.write_bytes(content)

    try:
        dataframe = read_dataset(file_path)
    except Exception as exc:  # pragma: no cover - rely on pandas exceptions
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data_store.set_dataset(file_path, dataframe)

    summary = summarize_dataframe(dataframe)
    return {
        "message": "File uploaded successfully.",
        "rows": int(len(dataframe)),
        "columns": dataframe.columns.tolist(),
        "summary": summary,
    }


@app.get("/columns")
async def get_columns() -> Dict[str, object]:
    dataframe = data_store.require_dataset()
    return {
        "columns": dataframe.columns.tolist(),
        "summary": summarize_dataframe(dataframe),
    }


@app.post("/train")
async def train(request: TrainRequest) -> Dict[str, object]:
    dataframe = data_store.require_dataset()
    try:
        results = train_models(
            dataframe,
            target_column=request.target_column,
            selected_models=request.algorithms,
            test_size=request.test_size or 0.2,
            random_state=request.random_state or 42,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data_store.set_results(request.target_column, results)
    return training_response_to_dict(results)


@app.get("/results")
async def get_results() -> Dict[str, object]:
    results = data_store.require_results()
    response = training_response_to_dict(results)
    response["target_column"] = data_store.target_column
    return response


@app.get("/download")
async def download_report(format: str = "csv") -> FileResponse:
    results = data_store.require_results()

    metrics = results.metrics
    if not metrics:
        raise HTTPException(status_code=400, detail="No metrics available for download.")

    extension = format.lower()
    if extension not in {"csv", "pdf"}:
        raise HTTPException(status_code=400, detail="Format must be either 'csv' or 'pdf'.")

    filename = f"model_comparison_{uuid.uuid4().hex}.{extension}"
    output_path = TEMP_DIR / filename

    if extension == "csv":
        export_metrics_csv(metrics, output_path)
        media_type = "text/csv"
    else:
        export_metrics_pdf(metrics, output_path)
        media_type = "application/pdf"

    return FileResponse(path=output_path, filename=filename, media_type=media_type)


