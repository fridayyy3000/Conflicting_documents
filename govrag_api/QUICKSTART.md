# GOV-RAG API - Quick Reference

## 🚀 Quick Start

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Authenticate with GCP (one time)
gcloud auth application-default login
gcloud auth application-default set-quota-project project-79920195-9e86-44ea-8c9

# 3. Start the API
./run.sh
```

## 📡 API Endpoints

**Base URL**: `http://localhost:8000`

### Health Check
```bash
curl http://localhost:8000/health
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the annual reimbursement ceiling for Tier-B employees at Velora Dynamics?"
  }'
```

### Batch Query
```bash
curl -X POST "http://localhost:8000/batch_query" \
  -H "Content-Type: application/json" \
  -d '["Question 1?", "Question 2?"]'
```

## 🧪 Testing

```bash
# Run test suite
python test_api.py

# Or test manually
curl http://localhost:8000/health
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| GCP auth errors | `gcloud auth application-default set-quota-project project-79920195-9e86-44ea-8c9` |
| Port in use | Change port in `main.py` or use `--port 8001` |
| Module not found | Make sure you're in the `govrag_api/` directory |

## 📁 Files

- **main.py** - FastAPI application
- **gov_rag_gemini.py** - Core GOV-RAG implementation
- **requirements.txt** - Python dependencies
- **run.sh** - Quick start script
- **test_api.py** - API test suite
- **easy/** - Document corpus (90 files)

## ✅ Status

- **Accuracy**: 100% on ConflictBench easy pack
- **Documents**: 90 fictional government documents (15 questions × 6 docs)
- **Model**: Gemini 2.5 Pro (Vertex AI)
- **Embeddings**: text-embedding-004
