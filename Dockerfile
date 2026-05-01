# Batch + API image for Cloud Run (linux/amd64).
#
# Build:
#   docker build --platform linux/amd64 -t llm-reranker:latest .
#
# Run batch (default CMD = main.py):
#   docker run --rm --env-file .env llm-reranker:latest
#   docker run --rm --env-file .env llm-reranker:latest main.py --limit 100
#
# Run HTTP API:
#   docker run --rm -p 8000:8000 --env-file .env llm-reranker:latest \
#     -m uvicorn api:app --host 0.0.0.0 --port 8000
#
# Cloud Run sets $PORT; use --port ${PORT:-8080} in the service command.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY utils/ utils/
COPY resources/ resources/
COPY main.py api.py ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["python"]
CMD ["main.py"]
