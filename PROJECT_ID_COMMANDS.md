# Quick Commands - Vertex AI with Project ID

## 🚀 Copy-Paste Commands (3 Steps)

### Step 1: Install Vertex AI
```bash
pip install google-cloud-aiplatform
```

### Step 2: Authenticate
```bash
gcloud auth application-default login
```

### Step 3: Set Project & Run
```bash
# Replace with YOUR project ID
export GOOGLE_CLOUD_PROJECT='your-project-id'

# Test setup
python test_vertex_setup.py

# Run evaluation
python code/evaluate_gov_rag_gemini.py --pack easy
```

---

## 📋 Complete Setup (if gcloud not installed)

```bash
# Install gcloud CLI (macOS)
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set default credentials
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Set environment variable
export GOOGLE_CLOUD_PROJECT='YOUR_PROJECT_ID'

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Verify
gcloud auth list
gcloud config get-value project
```

---

## 🔍 Find Your Project ID

```bash
# List all your projects
gcloud projects list

# Output shows:
# PROJECT_ID              NAME                PROJECT_NUMBER
# my-project-123456       My Project          123456789012
```

Use the **PROJECT_ID** column (e.g., `my-project-123456`)

---

## ▶️ Run GOV-RAG

```bash
# Set project (use YOUR project ID)
export GOOGLE_CLOUD_PROJECT='your-project-id'

# Easy pack
python code/evaluate_gov_rag_gemini.py --pack easy

# Hard pack  
python code/evaluate_gov_rag_gemini.py --pack hard

# Or pass project directly
python code/evaluate_gov_rag_gemini.py --pack easy --project-id your-project-id

# Interactive mode
python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy
```

---

## ✅ Test Your Setup

```bash
# Quick test
python test_vertex_setup.py

# Should show:
# ✓ google-cloud-aiplatform installed
# ✓ GOOGLE_CLOUD_PROJECT is set
# ✓ Authenticated as: your-email@gmail.com
# ✓ Vertex AI initialized
# ✓ Embedding model loaded
# ✓ Successfully generated embedding
```

---

## 🐛 Troubleshooting

### "gcloud not found"
```bash
brew install google-cloud-sdk
```

### "Could not find default credentials"
```bash
gcloud auth application-default login
```

### "Invalid project"
```bash
# List your projects to find the correct ID
gcloud projects list

# Set it
export GOOGLE_CLOUD_PROJECT='correct-project-id'
```

### "API not enabled"
```bash
gcloud services enable aiplatform.googleapis.com
```

### "Permission denied"
```bash
# Check IAM permissions in GCP Console
# You need: Vertex AI User role

# Or re-authenticate
gcloud auth application-default login
```

---

## 💡 Save Settings Permanently

Add to `~/.zshrc`:

```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
export VERTEX_AI_REGION='us-central1'
```

Then reload:
```bash
source ~/.zshrc
```

---

## 📊 Expected Results

```
================================================================================
GOV-RAG EVALUATION (VERTEX AI / GEMINI)
================================================================================
Pack: easy
Documents: conflictbench_fictional_full/packs/easy
Questions: conflictbench_fictional_full/questions.csv
Auth: Vertex AI (Project: your-project-id)
================================================================================

✓ Using Vertex AI
  Project: your-project-id
  Region: us-central1
  Model: text-embedding-004
  Auth: gcloud credentials

Loaded 180 documents
Creating embeddings for 180 documents...

[Processing questions...]

================================================================================
EVALUATION SUMMARY
================================================================================

Total Questions: 15

Answer Accuracy: 86.67% (13/15)
Gold Document Selection Rate: 80.00% (12/15)

Results saved to: results_gemini/
```

---

## 🎯 Your Next Command

```bash
# This single command should work if you have:
# 1. Installed: pip install google-cloud-aiplatform
# 2. Authenticated: gcloud auth application-default login
# 3. Set project: export GOOGLE_CLOUD_PROJECT='your-project-id'

python code/evaluate_gov_rag_gemini.py --pack easy
```
