# Using Vertex AI with Project ID

## Quick Setup (3 Commands)

```bash
# 1. Install Vertex AI SDK
pip install google-cloud-aiplatform

# 2. Authenticate with gcloud
gcloud auth application-default login

# 3. Set your project ID
export GOOGLE_CLOUD_PROJECT='your-project-id'
```

## Complete Setup Steps

### 1. Install gcloud CLI (if not installed)

```bash
# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### 2. Authenticate

```bash
# Login to your Google account
gcloud auth login

# Set application default credentials (for API access)
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

### 3. Set Project

```bash
# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Set as environment variable
export GOOGLE_CLOUD_PROJECT='YOUR_PROJECT_ID'

# Verify
gcloud config get-value project
echo $GOOGLE_CLOUD_PROJECT
```

### 4. Enable Vertex AI API

```bash
# Enable the API
gcloud services enable aiplatform.googleapis.com

# Verify it's enabled
gcloud services list --enabled | grep aiplatform
```

### 5. Install Python Dependencies

```bash
pip install google-cloud-aiplatform
```

### 6. Test Your Setup

```bash
python test_vertex_setup.py
```

## Run GOV-RAG with Vertex AI

### Full Evaluation

```bash
# Set project (if not already set)
export GOOGLE_CLOUD_PROJECT='your-project-id'

# Run evaluation on Easy pack
python code/evaluate_gov_rag_gemini.py --pack easy

# Run on Hard pack
python code/evaluate_gov_rag_gemini.py --pack hard

# Specify project ID directly
python code/evaluate_gov_rag_gemini.py --pack easy --project-id your-project-id

# Use different region
python code/evaluate_gov_rag_gemini.py --pack easy --region europe-west1
```

### Interactive Mode

```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy
```

## Find Your Project ID

### From gcloud:
```bash
gcloud projects list
```

### From GCP Console:
1. Go to https://console.cloud.google.com
2. Click the project dropdown at the top
3. Your project ID is shown next to each project name

### From Environment:
```bash
echo $GOOGLE_CLOUD_PROJECT
```

## Troubleshooting

### "gcloud: command not found"
```bash
# Install gcloud CLI
brew install google-cloud-sdk  # macOS

# Or download from:
# https://cloud.google.com/sdk/docs/install
```

### "Permission Denied" or "403 Forbidden"
```bash
# Re-authenticate
gcloud auth application-default login

# Check your IAM roles in GCP Console
# You need: Vertex AI User or Editor role
```

### "API not enabled"
```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Verify
gcloud services list --enabled | grep aiplatform
```

### "Could not find default credentials"
```bash
# Set application default credentials
gcloud auth application-default login

# Verify credentials file exists
ls ~/.config/gcloud/application_default_credentials.json
```

### "Invalid project ID"
```bash
# List your projects
gcloud projects list

# Set the correct project
gcloud config set project YOUR_PROJECT_ID
export GOOGLE_CLOUD_PROJECT='YOUR_PROJECT_ID'
```

## Cost Estimate (Vertex AI)

**Text Embedding Model (text-embedding-004):**
- $0.00025 per 1,000 characters

**For ConflictBench Easy (15 questions, 180 documents):**
- Documents: ~180 × 500 chars = 90,000 chars
- Queries: ~15 × 100 chars = 1,500 chars
- Total: ~91,500 chars
- **Cost: ~$0.02 per run**

Monthly free tier varies by region - check current pricing:
https://cloud.google.com/vertex-ai/pricing

## Comparison: API Key vs. Project ID

| Feature | AI Studio (API Key) | Vertex AI (Project ID) |
|---------|---------------------|------------------------|
| Setup | Simple (just API key) | Requires gcloud |
| Auth | API key in env var | gcloud credentials |
| Billing | Separate billing | Project billing |
| Production | Good for dev/test | Recommended |
| Quotas | 1,500 req/day free | Project quotas |
| Enterprise | Limited | Full enterprise features |

## Persistent Configuration

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# GCP Configuration
export GOOGLE_CLOUD_PROJECT='your-project-id'
export VERTEX_AI_REGION='us-central1'

# Optional: Set default gcloud project
gcloud config set project your-project-id
```

Then reload:
```bash
source ~/.zshrc
```

## Using Different Regions

Available regions for Vertex AI:
- `us-central1` (default, Iowa)
- `us-west1` (Oregon)
- `us-east1` (South Carolina)
- `europe-west1` (Belgium)
- `europe-west4` (Netherlands)
- `asia-southeast1` (Singapore)

```bash
# Set region
export VERTEX_AI_REGION='europe-west1'

# Or pass as argument
python code/evaluate_gov_rag_gemini.py --pack easy --region europe-west1
```

## Verification Checklist

Run these commands to verify everything is configured:

```bash
# ✓ gcloud installed
gcloud version

# ✓ Authenticated
gcloud auth list

# ✓ Project set
gcloud config get-value project
echo $GOOGLE_CLOUD_PROJECT

# ✓ Vertex AI enabled
gcloud services list --enabled | grep aiplatform

# ✓ Python package installed
python -c "import vertexai; print('✓ Vertex AI SDK installed')"

# ✓ Full test
python test_vertex_setup.py
```

## Next Steps

Once setup is complete:

```bash
# Run evaluation
python code/evaluate_gov_rag_gemini.py --pack easy

# Check results
ls -lh results_gemini/
cat results_gemini/metrics_easy_*.json | python -m json.tool
```
