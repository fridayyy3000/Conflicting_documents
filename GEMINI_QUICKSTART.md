# Quick Start with Gemini

## Prerequisites

1. **GCP Account** with billing enabled
2. **Gemini API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Setup (5 minutes)

### 1. Install Dependencies
```bash
# Install Gemini SDK
pip install google-generativeai

# Verify installation
python -c "import google.generativeai as genai; print('✓ Gemini SDK installed')"
```

### 2. Get API Key
```bash
# Go to: https://makersuite.google.com/app/apikey
# Create an API key
# Copy it to your terminal:

export GOOGLE_API_KEY='AIza...'  # Your actual API key

# Verify it's set
echo $GOOGLE_API_KEY
```

### 3. Test Gemini Connection
```bash
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
result = genai.embed_content(model='models/text-embedding-004', content='test')
print('✓ Gemini API connected successfully')
"
```

## Run GOV-RAG with Gemini

### Full Evaluation
```bash
# Easy pack (12 documents per question)
export GOOGLE_API_KEY='your-key-here'
python code/evaluate_gov_rag_gemini.py --pack easy

# Hard pack (20 documents per question)
python code/evaluate_gov_rag_gemini.py --pack hard

# Verbose mode
python code/evaluate_gov_rag_gemini.py --pack easy --verbose
```

### Interactive Mode
```bash
export GOOGLE_API_KEY='your-key-here'
python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy

# Then enter questions like:
# What is the maximum defect rate allowed for Q4-certified suppliers at Meridian Forge?
```

## Cost Estimate

**Gemini text-embedding-004 pricing:**
- Free tier: 1,500 requests/day
- After free tier: $0.00025 per 1,000 characters

**For ConflictBench Easy (15 questions, 180 documents):**
- Document embedding: ~180 documents × 500 chars = ~$0.02
- Query embedding: ~15 queries × 100 chars = ~$0.001
- **Total: ~$0.02 per run** (within free tier!)

## Advantages vs. Local Model

✅ **No model download** - works immediately  
✅ **No storage needed** - no 130MB model files  
✅ **Works behind firewalls** - just needs HTTPS to googleapis.com  
✅ **Latest embeddings** - Google's state-of-the-art models  
✅ **Scales easily** - handles large document sets

## Comparison: Local vs. Gemini

| Feature | Local (sentence-transformers) | Gemini API |
|---------|-------------------------------|------------|
| Setup Time | ~5 min (model download) | ~1 min |
| First Run | Slow (model loading) | Fast |
| Subsequent Runs | Fast (cached) | Fast |
| Cost | Free (compute only) | ~$0.02/run |
| Network Required | First time only | Every run |
| Works Offline | Yes (after download) | No |
| Embedding Quality | Good | Excellent |

## GCP Authentication (Advanced)

If you need more than the free tier or want production setup:

```bash
# Install gcloud CLI
brew install google-cloud-sdk

# Login
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable Vertex AI (if using Vertex AI instead of AI Studio)
gcloud services enable aiplatform.googleapis.com
```

## Troubleshooting

### "API key not set"
```bash
# Make sure you exported it
export GOOGLE_API_KEY='your-key-here'

# Check it's set
echo $GOOGLE_API_KEY

# Or pass directly to script
python code/evaluate_gov_rag_gemini.py --pack easy --api-key 'your-key-here'
```

### "Invalid API key"
- Get a new key from https://makersuite.google.com/app/apikey
- Make sure there are no quotes or spaces in the key
- Try regenerating the key

### "Quota exceeded"
- Free tier: 1,500 requests/day
- Wait 24 hours or enable billing
- Check quota at https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas

### "Connection error"
```bash
# Test basic connectivity
curl https://generativelanguage.googleapis.com

# Check firewall/proxy settings
# Gemini needs HTTPS access to generativelanguage.googleapis.com
```

## Running on GCP Compute Engine

```bash
# SSH to your VM
gcloud compute ssh your-instance-name

# Clone repo
git clone <your-repo>
cd Conflicting_documents

# Install dependencies
pip install -r requirements.txt
pip install google-generativeai

# Set API key
export GOOGLE_API_KEY='your-key-here'

# Run evaluation
python code/evaluate_gov_rag_gemini.py --pack easy
```

## Comparing Both Approaches

To compare local embeddings vs. Gemini:

```bash
# Run with local embeddings
python code/evaluate_gov_rag.py --pack easy

# Run with Gemini embeddings
export GOOGLE_API_KEY='your-key-here'
python code/evaluate_gov_rag_gemini.py --pack easy

# Compare results
ls -lh results/ results_gemini/
```

## Next Steps

Once you get results:
1. Check `results_gemini/` folder for detailed outputs
2. Compare metrics: answer accuracy, gold selection rate
3. Adjust authority weights in `code/gov_rag_gemini.py` if needed
4. Run ablation studies by modifying W_SEMANTIC, W_AUTHORITY, W_SCOPE
