# ML Classification Comparator

An end-to-end machine learning comparison tool built with FastAPI and Streamlit. Upload a structured dataset, select one or more popular classification algorithms, and compare evaluation metrics side-by-side to identify the best performing model.

## Features

- 📁 **Data upload:** Accepts CSV and Excel files, displays column metadata, and stores datasets locally.
- 🧹 **Automated preprocessing:** Handles missing values, encodes categorical features, and scales numeric data.
- 🤖 **Model zoo:** Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbours, Gaussian Naive Bayes, and XGBoost (when installed).
- 📊 **Evaluation dashboard:** Accuracy, Precision, Recall, F1-score, ROC-AUC, training time, and confusion matrices.
- 📥 **Report export:** Download comparison metrics as CSV or PDF.
- 🐳 **Docker-ready:** Single container runs both the FastAPI backend and the Streamlit frontend.

## Project structure

```
ml_comparator/
├── backend/
│   ├── main.py              # FastAPI application and endpoints
│   ├── model_trainer.py     # Training orchestration and model registry
│   └── utils/
│       ├── preprocess.py    # Data splitting and preprocessing utilities
│       ├── metrics.py       # Metric computation helpers
│       └── visualization.py # Export helpers for CSV/PDF reports
├── frontend/
│   ├── app.py               # Streamlit entry point
│   └── components/          # Modular UI building blocks
├── data/
│   ├── uploads/             # Uploaded datasets (gitignored)
│   └── temp/                # Generated reports (gitignored)
├── requirements.txt         # Python dependencies
└── Dockerfile               # Container definition
```

## Getting started

### Prerequisites

- Python 3.10+
- Node/JS is **not** required (Streamlit handles the frontend).

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ml_comparator/requirements.txt
```

### Run the backend

```bash
uvicorn ml_comparator.backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Run the frontend

In a new terminal session:

```bash
streamlit run ml_comparator/frontend/app.py
```

The Streamlit app defaults to the backend at `http://localhost:8000`. Update the sidebar field if you host the API elsewhere.

### Docker

Build and run both services in a single container:

```bash
docker build -t ml-classification-comparator ml_comparator/
docker run -p 8000:8000 -p 8501:8501 ml-classification-comparator
```

- FastAPI API: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

## API overview

| Method | Endpoint   | Description                     |
| ------ | ---------- | ------------------------------- |
| POST   | `/upload`  | Upload dataset and return summary |
| GET    | `/columns` | Retrieve detected columns and summary |
| POST   | `/train`   | Train selected algorithms        |
| GET    | `/results` | Fetch latest training results    |
| GET    | `/download`| Download CSV or PDF report       |

## Extending the project

- Add new algorithms by registering them in `MODEL_REGISTRY` inside `backend/model_trainer.py`.
- Add metrics or visuals in `backend/utils/metrics.py` and the Streamlit components.
- Integrate hyperparameter tuning by wrapping estimators with GridSearchCV or similar utilities.

## License

This project is provided as-is for demonstration purposes.
