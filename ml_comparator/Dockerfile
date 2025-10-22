FROM python:3.11-slim

WORKDIR /app

COPY ml_comparator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ml_comparator /app/ml_comparator

EXPOSE 8000 8501

ENV PYTHONPATH=/app

CMD ["bash", "-c", "uvicorn ml_comparator.backend.main:app --host 0.0.0.0 --port 8000 & streamlit run ml_comparator/frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
