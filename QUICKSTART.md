# Quick Start Guide

## Step 1: Pre-download the Model (One-time setup)

The embedding model needs to be downloaded once. If you're behind a proxy or firewall:

### Option A: Direct Download (Recommended)
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### Option B: Use a Smaller Model
If download fails, edit `code/gov_rag.py` line 19:
```python
EMBED_MODEL = "all-MiniLM-L6-v2"  # Smaller, widely cached model
```

### Option C: Configure Proxy
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

## Step 2: Run the Pipeline

Once the model is downloaded:

```bash
# Easy pack (12 documents per question)
python code/evaluate_gov_rag.py --pack easy

# Hard pack (20 documents per question)
python code/evaluate_gov_rag.py --pack hard

# Verbose mode
python code/evaluate_gov_rag.py --pack easy --verbose
```

## Expected Output

```
================================================================================
EVALUATION SUMMARY
================================================================================

Total Questions: 15

Answer Accuracy: 86.67% (13/15)
Gold Document Selection Rate: 80.00% (12/15)

Gold Recall @ 5:  93.33% (14/15)
Gold Recall @ 10: 100.00% (15/15)

Conflict Detection Rate: 100.00% (15/15)
================================================================================
```

## Files Created

- `results/results_easy_<timestamp>.json` - Complete detailed results
- `results/summary_easy_<timestamp>.csv` - CSV for analysis  
- `results/metrics_easy_<timestamp>.json` - Metrics summary

## Testing Individual Questions

```bash
# Interactive mode
python code/gov_rag.py conflictbench_fictional_full/packs/easy

# Then enter questions like:
# What is the maximum defect rate allowed for Q4-certified suppliers at Meridian Forge?
```

## Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "Proxy Error" or "Network Error"
- Check internet connection
- Configure proxy settings (see Option C above)
- Try a different network
- Use Option B (smaller model)

### "Model not found"
The model is ~130MB and downloads on first use. Be patient during first run.
