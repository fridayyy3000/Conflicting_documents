# GOV-RAG: Governing-Authority RAG for Conflicting Documents

End-to-end pipeline for evaluating RAG systems on conflicting-document benchmarks.

## Overview

GOV-RAG is designed to handle scenarios where multiple documents contain conflicting answers to the same question. The system:

1. **Broad Semantic Retrieval**: Retrieves top-30 candidates for high recall
2. **Claim Extraction**: Extracts factual claims from each document
3. **Authority Signal Detection**: Identifies document authority markers
4. **Conflict Detection**: Groups documents by their claimed answers
5. **Conflict Diversification**: Prevents duplicate claims from crowding context
6. **Authority-Aware Reranking**: Combines semantic relevance, authority, and scope
7. **Resolution**: Selects answer from most authoritative evidence

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

First run will download the embedding model (BAAI/bge-small-en-v1.5, ~130MB).

## Dataset Structure

```
conflictbench_fictional_full/
├── packs/
│   ├── easy/     # 12 documents per question (1 gold + 8 conflicts + 3 noise)
│   └── hard/     # 20 documents per question (1 gold + 14 conflicts + 5 noise)
├── questions.csv
└── ground_truth_manifest.csv
```

## Usage

### Run Full Evaluation

```bash
# Evaluate on Easy pack
python code/evaluate_gov_rag.py --pack easy

# Evaluate on Hard pack
python code/evaluate_gov_rag.py --pack hard

# Verbose mode (shows predictions during inference)
python code/evaluate_gov_rag.py --pack easy --verbose
```

### Interactive Mode

```bash
# Query documents interactively
python code/gov_rag.py conflictbench_fictional_full/packs/easy
```

## Evaluation Metrics

The evaluation script reports:

### Core Metrics
- **Answer Accuracy**: Percentage of correct answers
- **Gold Document Selection Rate**: Percentage of times the authoritative document was selected
- **Gold Recall @ K**: Whether gold document appears in top-K retrieved candidates (K=5,10,20,30)
- **Conflict Detection Rate**: Percentage of questions where conflicts were detected

### Diagnostic Metrics
- **Accuracy Given Gold Retrieved**: Conditional accuracy when gold is in top-30
- **Document Role Selection**: Breakdown of gold/conflict/noise document selections

## Output Files

Results are saved to `results/` directory:

- **`results_<pack>_<timestamp>.json`**: Complete results with candidates and scores
- **`summary_<pack>_<timestamp>.csv`**: CSV summary for analysis
- **`metrics_<pack>_<timestamp>.json`**: Metrics only

## Key Design Principles

### Strict Inference/Evaluation Separation

The pipeline ensures ground truth is **never visible during inference**:

1. **Inference Phase**: Only `question` and `question_id` are loaded from questions.csv
2. **Evaluation Phase**: After all predictions, ground truth is loaded for comparison

This prevents any leakage of gold answers or gold document identities into the inference process.

### Authority Signal Detection

For the fictional dataset, key authority markers:

- **Positive Signals**:
  - "ACTIVE AND AUTHORITATIVE"
  - "supersedes earlier"
  - "Policy Bulletin" (primary source)
  
- **Negative Signals**:
  - "should not be treated as the governing source"
  - "consult the active governing policy if a conflict exists"
  - "does not state the current requirement"
  - "Secondary internal summary"
  - "draft", "stale", "archived"

### Conflict Crowding Control

When multiple documents repeat the same wrong answer:
- Group documents by normalized claim
- Keep only the strongest representative per claim cluster
- Prevents majority voting from overwhelming the single gold document

## Configuration

Edit `code/gov_rag.py` to adjust:

```python
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # Embedding model
INITIAL_RETRIEVAL_K = 30                # Initial retrieval depth
FINAL_CONTEXT_K = 8                      # Final candidates after diversification

# Reranking weights
W_SEMANTIC = 0.30    # Semantic similarity weight
W_AUTHORITY = 0.50   # Authority score weight
W_SCOPE = 0.20       # Scope matching weight
```

## Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================

Total Questions: 15

Answer Accuracy: 86.67% (13/15)
Gold Document Selection Rate: 80.00% (12/15)

Gold Recall @ 5:  93.33% (14/15)
Gold Recall @ 10: 100.00% (15/15)
Gold Recall @ 20: 100.00% (15/15)
Gold Recall @ 30: 100.00% (15/15)

Conflict Detection Rate: 100.00% (15/15)
Accuracy (given gold retrieved): 86.67%

Noise Documents Selected: 1
Conflict Documents Selected: 2
Gold Documents Selected: 12
================================================================================
```

## Research Questions

The pipeline enables ablation studies:

1. **Semantic Only**: Set `W_AUTHORITY=0, W_SCOPE=0` → Pure embedding similarity
2. **Authority Only**: Set `W_SEMANTIC=0, W_SCOPE=0` → Pure authority scoring
3. **No Diversification**: Set `max_per_claim=999` → Allow duplicate claims
4. **Shallow Retrieval**: Set `INITIAL_RETRIEVAL_K=5` → Limited recall

Compare metrics across configurations to isolate the contribution of each component.

## Troubleshooting

### Import Errors
```bash
# Ensure you're running from repository root
cd /path/to/Conflicting_documents
python code/evaluate_gov_rag.py --pack easy
```

### Model Download Issues
```bash
# Test model download separately
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### Memory Issues
```bash
# Reduce batch size in sentence-transformers if needed
# Edit code/gov_rag.py: model.encode(..., batch_size=16)
```

## Citation

ConflictBench Fictional v1: A fully synthetic benchmark for conflicting-evidence retrieval and attribution.
