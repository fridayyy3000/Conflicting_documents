# Command Reference - GOV-RAG with Gemini

## 🚀 Quick Start (Copy-Paste Ready)

### 1. Install Gemini SDK
```bash
pip install google-generativeai
```

### 2. Get Your API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (starts with "AIza...")

### 3. Set Environment Variable
```bash
# Set your API key (replace with your actual key)
export GOOGLE_API_KEY='AIzaSy...'

# Verify it's set
echo $GOOGLE_API_KEY
```

### 4. Test Setup
```bash
python test_gemini_setup.py
```

### 5. Run Evaluation
```bash
# Easy pack (12 documents per question)
python code/evaluate_gov_rag_gemini.py --pack easy

# Hard pack (20 documents per question)
python code/evaluate_gov_rag_gemini.py --pack hard
```

---

## 📋 Complete Command Sequence

### For macOS/Linux:
```bash
# 1. Navigate to project
cd ~/Documents/Conflicting_documents

# 2. Install Gemini SDK
pip install google-generativeai

# 3. Set API key (get from https://makersuite.google.com/app/apikey)
export GOOGLE_API_KEY='AIzaSy...'

# 4. Test connection
python test_gemini_setup.py

# 5. Run evaluation on Easy pack
python code/evaluate_gov_rag_gemini.py --pack easy

# 6. Check results
ls -lh results_gemini/
cat results_gemini/metrics_easy_*.json
```

### For Windows (PowerShell):
```powershell
# 1. Navigate to project
cd C:\Users\YourName\Documents\Conflicting_documents

# 2. Install Gemini SDK
pip install google-generativeai

# 3. Set API key
$env:GOOGLE_API_KEY = "AIzaSy..."

# 4. Test connection
python test_gemini_setup.py

# 5. Run evaluation
python code\evaluate_gov_rag_gemini.py --pack easy

# 6. Check results
dir results_gemini\
type results_gemini\metrics_easy_*.json
```

---

## 🔑 GCP Authentication Commands

### Using AI Studio (Simpler - Recommended)
```bash
# Get API key from Google AI Studio
# https://makersuite.google.com/app/apikey

# Set environment variable
export GOOGLE_API_KEY='your-key-here'

# That's it! No gcloud needed for AI Studio
```

### Using Vertex AI (Advanced - for production)
```bash
# 1. Install gcloud CLI
brew install google-cloud-sdk  # macOS
# Or download from: https://cloud.google.com/sdk/docs/install

# 2. Login
gcloud auth login
gcloud auth application-default login

# 3. Set project
gcloud config set project YOUR_PROJECT_ID

# 4. Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com

# 5. Verify
gcloud auth list
gcloud config get-value project
```

---

## 📊 Running Different Configurations

### Basic Evaluation
```bash
export GOOGLE_API_KEY='your-key-here'
python code/evaluate_gov_rag_gemini.py --pack easy
```

### Verbose Mode (see predictions)
```bash
python code/evaluate_gov_rag_gemini.py --pack easy --verbose
```

### Custom Output Directory
```bash
python code/evaluate_gov_rag_gemini.py --pack easy --output-dir my_results
```

### Pass API Key Directly (no env var)
```bash
python code/evaluate_gov_rag_gemini.py --pack easy --api-key 'AIzaSy...'
```

### Run Both Easy and Hard
```bash
# Easy pack
python code/evaluate_gov_rag_gemini.py --pack easy

# Hard pack
python code/evaluate_gov_rag_gemini.py --pack hard

# Compare results
python -c "
import json
with open('results_gemini/metrics_easy_*.json') as f:
    easy = json.load(f)
with open('results_gemini/metrics_hard_*.json') as f:
    hard = json.load(f)
print(f'Easy Accuracy: {easy[\"answer_accuracy\"]:.2f}%')
print(f'Hard Accuracy: {hard[\"answer_accuracy\"]:.2f}%')
"
```

---

## 🧪 Interactive Testing

```bash
# Start interactive mode
export GOOGLE_API_KEY='your-key-here'
python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy

# Example questions to try:
# 1. What is the maximum defect rate allowed for Q4-certified suppliers at Meridian Forge?
# 2. What is the annual reimbursement ceiling for Tier-B employees at Velora Dynamics?
# 3. How long must Tier-3 diagnostic records be retained at Arclume Health?

# Type 'exit' to quit
```

---

## 🔧 Troubleshooting Commands

### Check if API key is set
```bash
echo $GOOGLE_API_KEY
# Should show: AIzaSy... (not empty)
```

### Test Gemini connection
```bash
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
result = genai.embed_content(model='models/text-embedding-004', content='test')
print('✓ Connected! Embedding dimension:', len(result['embedding']))
"
```

### Reinstall Gemini SDK
```bash
pip uninstall google-generativeai -y
pip install google-generativeai
```

### Check installed version
```bash
pip show google-generativeai
```

### Test with sample embedding
```bash
python -c "
import google.generativeai as genai
genai.configure(api_key='YOUR_KEY_HERE')
texts = ['What is X?', 'What is Y?']
result = genai.embed_content(model='models/text-embedding-004', content=texts)
print(f'✓ Embedded {len(texts)} texts successfully')
"
```

---

## 📁 Check Results

### View metrics
```bash
# Latest metrics
cat results_gemini/metrics_easy_*.json | python -m json.tool

# Extract key metrics
python -c "
import json, glob
f = glob.glob('results_gemini/metrics_easy_*.json')[-1]
m = json.load(open(f))
print(f'Answer Accuracy: {m[\"answer_accuracy\"]:.2f}%')
print(f'Gold Selection: {m[\"gold_selection_rate\"]:.2f}%')
print(f'Recall@10: {m[\"gold_recall_at_10\"]:.2f}%')
"
```

### View CSV summary
```bash
# Open in spreadsheet
open results_gemini/summary_easy_*.csv

# Or view in terminal
cat results_gemini/summary_easy_*.csv | column -t -s,
```

### View detailed results
```bash
# Pretty print JSON
cat results_gemini/results_easy_*.json | python -m json.tool | less
```

---

## 💰 Check API Usage

### Free tier limits
```bash
# Check quota usage (requires gcloud)
gcloud alpha services quota get \
  --service=generativelanguage.googleapis.com \
  --consumer=projects/YOUR_PROJECT

# Or visit: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
```

### Estimate cost for your run
```bash
python -c "
# ConflictBench Easy: 15 questions, 180 documents
docs = 180
queries = 15
avg_doc_chars = 500
avg_query_chars = 100

total_chars = (docs * avg_doc_chars) + (queries * avg_query_chars)
cost = (total_chars / 1000) * 0.00025  # Gemini pricing

print(f'Total characters: {total_chars:,}')
print(f'Estimated cost: \${cost:.4f}')
print(f'Free tier: {1500 - (docs + queries)} requests remaining')
"
```

---

## 🔄 Compare Local vs. Gemini

```bash
# Run with local embeddings (sentence-transformers)
python code/evaluate_gov_rag.py --pack easy

# Run with Gemini embeddings
export GOOGLE_API_KEY='your-key-here'
python code/evaluate_gov_rag_gemini.py --pack easy

# Compare metrics
diff -y \
  <(cat results/metrics_easy_*.json | python -m json.tool) \
  <(cat results_gemini/metrics_easy_*.json | python -m json.tool)
```

---

## 🎓 Advanced: Running on GCP VM

```bash
# 1. Create VM
gcloud compute instances create govrag-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

# 2. SSH to VM
gcloud compute ssh govrag-vm --zone=us-central1-a

# 3. Setup on VM
sudo apt-get update
sudo apt-get install -y python3-pip git
git clone <your-repo-url>
cd Conflicting_documents
pip3 install -r requirements.txt
pip3 install google-generativeai

# 4. Set API key
export GOOGLE_API_KEY='your-key-here'

# 5. Run evaluation
python3 code/evaluate_gov_rag_gemini.py --pack easy

# 6. Download results
exit  # exit SSH
gcloud compute scp govrag-vm:~/Conflicting_documents/results_gemini/* ./results_gemini/ --zone=us-central1-a
```

---

## 📝 Save Commands for Later

```bash
# Add to your ~/.zshrc or ~/.bashrc for persistence
echo "export GOOGLE_API_KEY='your-key-here'" >> ~/.zshrc
source ~/.zshrc

# Or create a .env file
cat > .env << EOF
GOOGLE_API_KEY=your-key-here
EOF

# Load it before running
source .env
python code/evaluate_gov_rag_gemini.py --pack easy
```

---

## ✅ Quick Health Check

```bash
# All-in-one health check
python -c "
import os
import sys

print('Checking setup...')
checks = []

# Check 1: API key
checks.append(('API Key', bool(os.getenv('GOOGLE_API_KEY'))))

# Check 2: Package
try:
    import google.generativeai
    checks.append(('Gemini SDK', True))
except:
    checks.append(('Gemini SDK', False))

# Check 3: Files
import pathlib
checks.append(('Code files', 
    pathlib.Path('code/gov_rag_gemini.py').exists()))
checks.append(('Documents', 
    pathlib.Path('conflictbench_fictional_full/packs/easy').exists()))

for name, status in checks:
    print(f'  {\"✓\" if status else \"✗\"} {name}')

if all(c[1] for c in checks):
    print('\\n✓ All systems ready! Run:')
    print('  python code/evaluate_gov_rag_gemini.py --pack easy')
else:
    print('\\n✗ Some checks failed. See GEMINI_QUICKSTART.md')
"
```
