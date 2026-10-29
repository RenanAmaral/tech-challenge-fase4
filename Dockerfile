FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e . && python scripts/train_models.py
EXPOSE 8000 8501
CMD ["uvicorn", "hospital_monitor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
