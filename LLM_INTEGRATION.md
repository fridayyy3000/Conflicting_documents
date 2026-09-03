# GOV-RAG with Gemini 2.5 Pro Integration

## 🎯 What Changed

Your GOV-RAG pipeline now includes **Gemini 2.5 Pro** for final evidence resolution!

### Updated Pipeline

```
Question
   ↓
Broad semantic retrieval (top 20–30)
   ↓
Claim extraction
   ↓
Conflict grouping
   ↓
Authority + scope scoring
   ↓
Conflict-crowding control
   ↓
Top governing candidates (8 documents)
   ↓
✨ GEMINI 2.5 PRO ✨  ← NEW!
   ↓
Final answer + governing source
```

---

## 📝 The Prompt

Gemini 2.5 Pro uses this system prompt for evidence resolution:

```
You are the final evidence-resolution component of GOV-RAG, a retrieval system
designed for question answering over conflicting documents.

You must answer the user's question using ONLY the retrieved evidence provided to you.

The retrieved documents may intentionally disagree.

Your task is not to choose the most frequently repeated answer.
Your task is to identify the answer supported by the governing source that applies
to the exact scope of the question.

When resolving conflicts, consider:

1. Exact scope: Prefer documents that apply to the exact entity/category asked about
2. Authority: Prefer active governing policies over FAQs, summaries, training guides
3. Current status: Prefer active/current documents over archived/superseded
4. Supersession: If one document supersedes another, prefer the superseding document
5. Directness: Prefer documents that directly state the requirement
6. Evidence frequency is NOT authority: Multiple secondary documents repeating 
   the same value do not outweigh one clearly authoritative governing document

Return JSON only in this format:
{
  "answer": "<final answer>",
  "selected_source": "<filename>",
  "reason": "<brief explanation of why this source governs>",
  "conflict_detected": true,
  "confidence": "high|medium|low"
}
```

---

## 🚀 How to Use

### Option 1: Evaluation Script (with LLM)

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT='project-79920195-9e86-44ea-8c9'

# Authenticate
gcloud auth application-default login

# Run evaluation with Gemini 2.5 Pro
python code/evaluate_gov_rag_gemini.py --pack easy
```

By default, **`use_llm=True`** is enabled, so Gemini will be used for final answer generation.

### Option 2: Interactive Mode (with LLM)

```bash
export GOOGLE_CLOUD_PROJECT='project-79920195-9e86-44ea-8c9'

python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy
```

Then type your questions interactively.

---

## 🔧 Advanced Options

### Disable LLM (Use Rule-Based Resolution Only)

If you want to test the original rule-based authority resolution without LLM:

```python
from gov_rag_gemini import GovRAGGemini

# Initialize with use_llm=False
rag = GovRAGGemini(
    'conflictbench_fictional_full/packs/easy',
    project_id='your-project-id',
    use_llm=False  # Disable LLM generation
)

result = rag.query("What is the maximum speed limit?")
```

### Compare LLM vs Rule-Based

```bash
# Run with LLM (default)
python code/evaluate_gov_rag_gemini.py --pack easy --output-dir results_llm

# Run without LLM (add --no-llm flag if we add it)
# For now, you'd need to modify the code to set use_llm=False
```

---

## 📊 Output Format

### With LLM Enabled

```json
{
  "answer": "100 km/h",
  "source": "Q001_source_01.md",
  "reason": "This document is marked as ACTIVE AND AUTHORITATIVE and explicitly states the current requirement for Category A vehicles",
  "conflict_detected": true,
  "confidence": "high",
  "method": "llm",
  "candidates": [...]
}
```

### Without LLM (Rule-Based)

```json
{
  "answer": "100 km/h",
  "source": "Q001_source_01.md",
  "supporting_sentence": "The maximum speed limit is 100 km/h",
  "status": "active_authoritative",
  "document_type": "governing_policy",
  "score": 0.856,
  "method": "rule-based",
  "candidates": [...]
}
```

---

## 🔍 What Gemini Sees

For each question, Gemini receives:

1. **System Prompt**: Instructions on how to resolve conflicts
2. **User Prompt**: 
   - The question
   - Top 8 retrieved documents with:
     - Filename
     - Status (active/superseded/archived/draft)
     - Type (governing_policy/faq/training_guide/etc)
     - Authority score
     - Scope score
     - Document text (first 1500 chars)

Example input to Gemini:

```
Question: What is the maximum speed limit for Category A vehicles?

Retrieved Evidence:

Document 1:
Filename: Q001_source_01.md
Status: active_authoritative
Type: governing_policy
Authority Score: 0.900
Scope Score: 0.856

Content:
# Speed Limit Policy for Category A Vehicles
Status: ACTIVE AND AUTHORITATIVE
...

Document 2:
Filename: Q001_source_08.md
Status: superseded
Type: training_guide
Authority Score: 0.320
Scope Score: 0.745

Content:
# Training Guide: Speed Limits
Note: This is a training guide from 2018...
```

---

## 💰 Cost Estimate

### Gemini 2.5 Pro (Default)

- **Input**: ~1,500 tokens per question (8 docs × ~150 tokens + prompt)
- **Output**: ~100 tokens per answer
- **Cost per Question**: ~$0.005 (Gemini 2.5 Pro pricing)
- **Cost for Easy Pack** (15 questions): ~$0.075
- **Cost for Hard Pack** (15 questions): ~$0.075

### Alternative: Gemini 2.0 Flash (Faster/Cheaper)

- **Cost per Question**: ~$0.0003
- **Cost for Easy Pack**: ~$0.005

To use Gemini 2.0 Flash instead, edit `code/gov_rag_gemini.py`:

```python
VERTEX_AI_GENERATION_MODEL = "gemini-2.0-flash-exp"  # Change from gemini-2.5-pro
```

---

## ✅ Verification

To verify the LLM integration is working:

```bash
# Test Vertex AI connection
python test_vertex_setup.py

# Run a quick test
export GOOGLE_CLOUD_PROJECT='project-79920195-9e86-44ea-8c9'
python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy

# Type a question and look for:
# "FINAL RESOLUTION: Gemini 2.5 Pro"
# in the output
```

---

## 🐛 Troubleshooting

### "FINAL RESOLUTION: Gemini 2.5 Pro" not appearing

Check that `use_llm=True` in the code. By default it should be enabled.

### "Gemini generation failed"

1. Check authentication:
   ```bash
   gcloud auth application-default print-access-token
   ```

2. Verify project ID:
   ```bash
   echo $GOOGLE_CLOUD_PROJECT
   ```

3. Enable Vertex AI API:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

### Fallback to rule-based resolution

If Gemini fails, the system automatically falls back to the original rule-based authority resolution. You'll see:

```
ERROR: Gemini generation failed: <error message>
```

And the result will have `"method": "rule-based"` instead of `"method": "llm"`.

---

## 📁 Modified Files

1. **`code/gov_rag_gemini.py`**
   - Added `SYSTEM_PROMPT` constant with evidence resolution instructions
   - Added `GeminiGenerator` class for LLM-based answer generation
   - Modified `GovRAGGemini.__init__()` to accept `use_llm` parameter
   - Modified `GovRAGGemini.query()` to use Gemini for final resolution
   - Added `_format_evidence()` to prepare documents for LLM
   - Added `_fallback_resolution()` for graceful degradation

2. **`code/evaluate_gov_rag_gemini.py`**
   - Updated to pass `use_llm=True` when initializing GOV-RAG
   - Fixed path construction bugs
   - Added support for `--project-id` argument

---

## 🎯 Next Steps

1. **Run the auth fix script**:
   ```bash
   ./fix_gcp_auth.sh
   ```

2. **Test the setup**:
   ```bash
   python test_vertex_setup.py
   ```

3. **Run evaluation with LLM**:
   ```bash
   export GOOGLE_CLOUD_PROJECT='project-79920195-9e86-44ea-8c9'
   python code/evaluate_gov_rag_gemini.py --pack easy
   ```

4. **Compare with rule-based**:
   - Look at the JSON output
   - Check `"method": "llm"` vs `"method": "rule-based"`
   - Compare `"reason"` field (LLM explains why it chose that source)
   - Compare `"confidence"` levels

---

## 📊 Expected Improvements

With LLM-based resolution, you should see:

1. **Better Conflict Resolution**: LLM can reason about multiple documents simultaneously
2. **Clearer Explanations**: The `"reason"` field explains why a source was chosen
3. **Confidence Scores**: High/medium/low confidence based on evidence quality
4. **Conflict Detection**: Automatically detects when documents disagree
5. **Scope Awareness**: Better understanding of which document applies to the question's exact scope

The rule-based system is still excellent and may be faster/cheaper for some use cases. The LLM adds an extra layer of reasoning that can help with edge cases.
