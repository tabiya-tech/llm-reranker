# Batch job image for Cloud Run (linux/amd64).
# Reads ClassifiedJobs from MongoDB, runs Gemini LLM pipeline, writes RankedJobs.
#
# Build:
#   docker build --platform linux/amd64 -t llm-reranker:latest .
#
# Run (one-shot batch):
#   docker run --rm --env-file .env llm-reranker:latest
#
# Run (API mode):
#   docker run --rm -p 8000:8000 --env-file .env llm-reranker:latest api

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

ENTRYPOINT ["python", "main.py"]
