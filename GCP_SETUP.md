# GCP Setup for GOV-RAG

## 1. Install Google Cloud SDK

```bash
# If not already installed
# macOS:
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

## 2. Authenticate with GCP

### Option A: User Authentication (Development)
```bash
# Login to your Google account
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Application default credentials (for API access)
gcloud auth application-default login
```

### Option B: Service Account (Production)
```bash
# Download service account key from GCP Console
# Then set the environment variable:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Verify authentication
gcloud auth list
```

## 3. Enable Required APIs

```bash
# Enable Vertex AI API (for Gemini)
gcloud services enable aiplatform.googleapis.com

# Check enabled services
gcloud services list --enabled
```

## 4. Install Python Dependencies

```bash
# Install Google AI Python SDK
pip install google-generativeai

# Or for Vertex AI
pip install google-cloud-aiplatform
```

## 5. Verify Setup

```bash
# Check authentication
gcloud auth list

# Check project
gcloud config get-value project

# Test Vertex AI access
gcloud ai models list --region=us-central1
```

## Using Gemini 2.5 Pro in GOV-RAG

### Option 1: Use Gemini for Embeddings

The GOV-RAG pipeline currently uses sentence-transformers for embeddings. You can switch to Gemini's embedding API to avoid downloading models.

See `code/gov_rag_gemini.py` for implementation.

### Option 2: Use Gemini for Claim Extraction (Advanced)

The current pipeline uses regex for claim extraction. You could enhance this with Gemini for better natural language understanding.

### Option 3: Keep Current Architecture

**Recommended**: The current GOV-RAG pipeline doesn't need an LLM. It uses:
- Local embeddings (sentence-transformers)
- Rule-based authority detection
- Regex-based claim extraction

This is by design - it's a training-free, interpretable system.

## Gemini API Pricing (as of 2024)

- **Gemini 2.5 Pro**: 
  - Input: $1.25 / 1M tokens
  - Output: $5.00 / 1M tokens
  
- **Gemini 2.5 Flash**:
  - Input: $0.075 / 1M tokens
  - Output: $0.30 / 1M tokens

For 15 questions with 180 documents (Easy pack), using Gemini for embeddings would cost ~$0.10-0.50 depending on approach.

## Quick Commands Reference

```bash
# Login
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Set region (optional)
gcloud config set ai/region us-central1

# List available models
gcloud ai models list --region=us-central1 | grep gemini

# Test Gemini API
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_API_KEY'); print('Gemini API ready')"
```

## Environment Variables

```bash
# Add to ~/.zshrc or ~/.bashrc
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_API_KEY="your-api-key"  # If using generative AI SDK
```

## Troubleshooting

### "Permission Denied" Error
```bash
# Re-authenticate
gcloud auth application-default login

# Check your IAM roles in GCP Console
# Need: Vertex AI User or Vertex AI Administrator
```

### "Project not set" Error
```bash
gcloud config set project YOUR_PROJECT_ID
```

### "API not enabled" Error
```bash
gcloud services enable aiplatform.googleapis.com
```
