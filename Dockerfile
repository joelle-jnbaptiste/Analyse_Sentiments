FROM python:3.9-slim

RUN apt-get update && apt-get clean && apt-get install -y ca-certificates

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=3

COPY ./requirements-api.txt /app/requirements-api.txt

COPY model_final /app/model_final/

COPY api /app/api/

RUN pip install --no-cache-dir -r requirements-api.txt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4" ]
