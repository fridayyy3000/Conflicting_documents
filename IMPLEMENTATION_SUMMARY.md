# GOV-RAG Pipeline Implementation Summary

## ✅ Complete End-to-End Pipeline Created

### Core Components

1. **`code/gov_rag.py`** - Main GOV-RAG implementation
   - Semantic retrieval (top-30 broad recall)
   - Authority signal detection (specific to fictional dataset)
   - Claim extraction (numeric/quantitative values)
   - Conflict detection and grouping
   - Duplicate-claim diversification (prevents crowding)
   - Authority-aware reranking
   - Resolution logic

2. **`code/evaluate_gov_rag.py`** - Evaluation pipeline
   - **Strict separation of inference and evaluation**
   - Loads only questions during inference (no gold answers/documents)
   - Runs GOV-RAG on all questions in pack
   - Loads ground truth AFTER inference complete
   - Calculates comprehensive metrics
   - Saves detailed results

3. **Documentation**
   - `QUICKSTART.md` - Quick start instructions
   - `GOVRAG_README.md` - Complete documentation
   - `requirements.txt` - Dependencies

---

## 🎯 Key Requirements Addressed

### ✅ Dataset Handling
- **Correct path**: Uses `conflictbench_fictional_full/packs/easy` and `packs/hard`
- **Fictional documents**: All entities, companies, policies are synthetic
- **Questions from CSV**: Loads from `conflictbench_fictional_full/questions.csv`

### ✅ Inference/Evaluation Separation
**CRITICAL: Ground truth is NEVER visible during inference**

```python
# PHASE 1: Inference (NO ground truth)
questions = load_questions_for_inference(csv)  # Only question_id + question

# PHASE 2: Prediction
predictions = rag.query(question)  # No access to gold_answer or gold_document

# PHASE 3: Evaluation (AFTER all predictions)
gold_data = load_ground_truth(csv)  # Now load answers/sources
evaluate(predictions, gold_data)
```

### ✅ Authority Detection (Fictional Dataset)

**Positive Signals (+):**
- "ACTIVE AND AUTHORITATIVE" → +10.0
- "supersedes earlier" → +4.0
- "governing status: active" → +8.0
- "Policy Bulletin" → +2.0

**Negative Signals (-):**
- "should not be treated as the governing source" → -8.0
- "consult the active governing policy if a conflict exists" → -6.0
- "does not state the current requirement" → -7.0
- "Secondary internal summary" → -4.0
- "Background Note" → -3.0
- "draft" → -4.0, "stale" → -5.0, "archived" → -5.0

### ✅ Conflict Crowding Control

Example: If 5 documents claim "1.00%" but 1 gold document claims "0.72%":

**Without diversification:**
```
Top-8: [1.00%, 1.00%, 1.00%, 1.00%, 1.00%, 0.85%, 0.50%, 1.25%]
       Gold (0.72%) is crowded out!
```

**With diversification:**
```
Top-8: [0.72%, 1.00%, 0.85%, 0.50%, 1.25%, ...]
       Gold preserved, distinct answers represented
```

Implementation:
```python
# Group by normalized claim
groups = group_by_claim(candidates)

# Keep only strongest representative per claim
for claim, docs in groups.items():
    docs.sort(by=final_score)
    keep_top_1_per_claim(docs)
```

### ✅ Comprehensive Metrics

**Core Metrics:**
- Answer Accuracy (correct answer %)
- Gold Document Selection Rate
- Gold Recall @ 5, 10, 20, 30
- Conflict Detection Rate

**Diagnostic Metrics:**
- Accuracy given gold was retrieved
- Document role selection breakdown (gold/conflict/noise)

**Ablation Support:**
Configure weights to test components:
```python
# Semantic only
W_SEMANTIC=1.0, W_AUTHORITY=0.0, W_SCOPE=0.0

# Authority only  
W_SEMANTIC=0.0, W_AUTHORITY=1.0, W_SCOPE=0.0

# Balanced (current)
W_SEMANTIC=0.30, W_AUTHORITY=0.50, W_SCOPE=0.20
```

---

## 📁 Output Files

Every run produces:

1. **`results/results_<pack>_<timestamp>.json`**
   - Complete results with all candidates
   - Authority scores, semantic scores, scope scores
   - Final ranking for each question
   - Supporting sentences

2. **`results/summary_<pack>_<timestamp>.csv`**
   - One row per question
   - Predicted vs. gold answers
   - Selected vs. gold documents
   - Recall metrics per question
   - Easy to analyze in spreadsheet

3. **`results/metrics_<pack>_<timestamp>.json`**
   - Aggregated metrics only
   - Quick performance summary

---

## 🚀 Usage

### Full Evaluation
```bash
# Install dependencies (one-time)
pip install -r requirements.txt

# Run evaluation on Easy pack
python code/evaluate_gov_rag.py --pack easy

# Run evaluation on Hard pack
python code/evaluate_gov_rag.py --pack hard

# Verbose mode (see predictions during inference)
python code/evaluate_gov_rag.py --pack easy --verbose
```

### Interactive Testing
```bash
# Query documents interactively
python code/gov_rag.py conflictbench_fictional_full/packs/easy
```

---

## 📊 Expected Performance

On Easy pack (12 docs/question):
- Gold Recall @ 30 should be ~100% (semantic retrieval working)
- Gold Selection Rate: Target 70-90% (authority detection working)
- Answer Accuracy: Should match gold selection (if claim extraction works)

On Hard pack (20 docs/question):
- More challenging due to more conflicts
- Gold recall should remain high
- Selection accuracy tests authority scoring strength

---

## 🔧 Configuration

Edit `code/gov_rag.py` to adjust:

```python
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # Change if download fails
INITIAL_RETRIEVAL_K = 30                # Increase for more recall
FINAL_CONTEXT_K = 8                      # Top-K after diversification

W_SEMANTIC = 0.30    # Semantic similarity weight
W_AUTHORITY = 0.50   # Authority score weight  
W_SCOPE = 0.20       # Scope matching weight
```

---

## ⚠️ Current Limitation

**Model Download Required:**
The embedding model (BAAI/bge-small-en-v1.5, ~130MB) downloads on first run.

If you encounter network/proxy errors:

1. **Try a different network** (not behind corporate proxy)
2. **Use a smaller model**: Edit line 19 in `gov_rag.py`:
   ```python
   EMBED_MODEL = "all-MiniLM-L6-v2"
   ```
3. **Pre-download manually** (on a machine with internet):
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
   ```
   Then copy `~/.cache/huggingface/hub/` to the target machine

---

## ✅ Research Design Principles Implemented

1. **No Ground Truth Leakage**
   - Inference sees only questions
   - Gold answers/documents loaded AFTER predictions
   
2. **Modular Architecture**
   - Each stage is separate function
   - Easy to ablate components
   
3. **Reproducible**
   - All scores saved
   - All intermediate candidates preserved
   
4. **Transparent**
   - Authority scores visible
   - Semantic scores visible  
   - Final ranking explained
   
5. **Conflict-Aware**
   - Detects conflicting claims
   - Groups by answer
   - Controls duplicate crowding

---

## 🎓 Research Questions Enabled

1. **Does authority scoring help?**
   - Compare W_AUTHORITY=0.5 vs W_AUTHORITY=0.0
   
2. **Does conflict diversification help?**
   - Compare max_per_claim=1 vs max_per_claim=999
   
3. **What's the optimal retrieval depth?**
   - Compare INITIAL_RETRIEVAL_K=10,20,30,40
   
4. **Do we need all three signals?**
   - Test semantic-only, authority-only, scope-only
   
5. **How does performance degrade with conflict density?**
   - Compare Easy (8 conflicts) vs Hard (14 conflicts)

---

## 📝 Next Steps

1. **Resolve model download** (see troubleshooting in QUICKSTART.md)
2. **Run first evaluation**: `python code/evaluate_gov_rag.py --pack easy`
3. **Inspect results**: Check `results/` directory
4. **Iterate**:
   - Adjust authority weights if gold selection is low
   - Adjust retrieval depth if gold recall is low
   - Review failed cases in detailed results JSON

---

## Summary

✅ Complete end-to-end pipeline  
✅ Strict inference/evaluation separation  
✅ Authority detection for fictional dataset  
✅ Conflict crowding control  
✅ Comprehensive evaluation metrics  
✅ Modular design for ablations  
✅ Detailed documentation  

**Ready to run!** (after resolving model download)
