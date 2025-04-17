![Deploy to Azure](https://github.com/joelle-jnbaptiste/Analyse_Sentiments/actions/workflows/deploy.yml/badge.svg)

# Sentiment Analysis API

This project contains a sentiment analysis API using FastAPI and a TensorFlow Lite model (DistilBERT).  
It includes model loading, prediction logic, feedback logging, and unit tests.

## Project Structure

```
ANALYSE_SENTIMENTS/
│
├── api/                     # FastAPI app
│   └── main.py
│
├── model_final/             # Model loading logic
│   ├── DISTILBERT_MODEL_TFLITE/
│   ├── distilbert_model.tflite
│   └── model_loader.py
│
├── tests/                   # Unit tests
│   └── test_model.py
│
├── Dockerfile               # Containerization for API
├── requirements.txt         # Core dependencies
├── requirements-api.txt     # API-specific dependencies
├── pytest.ini               # Pytest configuration
```

---

## 🚀 Run with Docker

> ❗ Dockerfile details depend on your current setup. Please refer to your `Dockerfile`.

### 1. Build the Docker image

```bash
docker build -t sentiment-api .
```

### 2. Run the container locally

```bash
docker run -p 8000:8000 sentiment-api
```

Then access the API at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ✅ Available Endpoints

- `POST /predict` — Predict sentiment from a text
- `POST /feedback` — Send user feedback on prediction

---

## 🧪 Test Examples

Tests in `tests/test_model.py` include:

- Basic positive/negative prediction tests
- Batch accuracy verification (≥80%)

---
