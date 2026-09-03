FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY gov_rag_gemini.py main.py corpus_store.py document_parser.py rate_limit.py chunker.py ./
COPY easy/ ./easy/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Cloud Run injects $PORT, so use shell form to expand it.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
