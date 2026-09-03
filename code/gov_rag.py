# gov_rag.py

import os
import re
import json
import glob
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

# Document directory is now passed as parameter
# DOC_DIR will be set at runtime based on --pack argument

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

INITIAL_RETRIEVAL_K = 30
FINAL_CONTEXT_K = 8

# Reranking weights
W_SEMANTIC = 0.30
W_AUTHORITY = 0.50
W_SCOPE = 0.20


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    filename: str
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Candidate:
    filename: str
    text: str

    semantic_score: float
    authority_score: float
    scope_score: float

    claim: Optional[str]
    claim_sentence: Optional[str]

    status: str
    doc_type: str

    final_score: float = 0.0


# ============================================================
# EMBEDDINGS
# ============================================================

class Embedder:
    def __init__(self, model_name=EMBED_MODEL):
        """
        Initialize embedder with specified model.
        
        If model download fails (network issues), provides helpful error message.
        """
        try:
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"\nERROR: Failed to load embedding model '{model_name}'")
            print(f"Reason: {e}")
            print("\n" + "="*80)
            print("TROUBLESHOOTING:")
            print("="*80)
            print("\n1. Check your internet connection")
            print("\n2. If behind a proxy, configure it:")
            print("   export HTTP_PROXY=http://proxy:port")
            print("   export HTTPS_PROXY=http://proxy:port")
            print("\n3. Pre-download the model using:")
            print(f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\"")
            print("\n4. Or use a different model by editing EMBED_MODEL in gov_rag.py")
            print("   Options: 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'")
            print("="*80 + "\n")
            raise

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(folder: str) -> List[Document]:

    files = glob.glob(os.path.join(folder, "**/*.md"), recursive=True)

    docs = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        docs.append(
            Document(
                filename=os.path.basename(path),
                text=text
            )
        )

    print(f"Loaded {len(docs)} documents")

    return docs


# ============================================================
# BASIC TEXT UTILITIES
# ============================================================

def cosine_similarity(a, b):
    return float(np.dot(a, b))


def sentences(text):
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+|\n+", text)
        if x.strip()
    ]


# ============================================================
# AUTHORITY DETECTION
# ============================================================

def detect_status(text: str):

    t = text.lower()

    # Check for explicit authoritative markers (fictional dataset)
    if "active and authoritative" in t:
        return "active_authoritative"

    if "governing status: active" in t:
        return "active_authoritative"

    if any(x in t for x in [
        "superseded",
        "no longer valid",
        "withdrawn",
        "obsolete",
        "superseded by"
    ]):
        return "superseded"

    if any(x in t for x in [
        "archived",
        "historical",
        "legacy version",
        "stale"
    ]):
        return "archived"

    if any(x in t for x in [
        "draft",
        "proposed",
        "working draft",
        "planning brief"
    ]):
        return "draft"

    if any(x in t for x in [
        "active governing policy",
        "governing policy",
        "currently effective",
        "current policy",
        "effective immediately",
        "active policy"
    ]):
        return "active"

    return "unknown"


def detect_doc_type(text: str):

    t = text.lower()

    hierarchy = [
        ("governing_policy", [
            "governing policy",
            "official policy"
        ]),

        ("policy_bulletin", [
            "policy bulletin"
        ]),

        ("standard", [
            "official standard",
            "safety standard",
            "compliance standard"
        ]),

        ("implementation_memo", [
            "implementation memo",
            "implementation memorandum"
        ]),

        ("faq", [
            "faq",
            "frequently asked questions"
        ]),

        ("training_guide", [
            "training guide",
            "training material"
        ]),

        ("reference_card", [
            "reference card",
            "quick reference"
        ]),

        ("secondary_summary", [
            "summary",
            "digest",
            "overview"
        ])
    ]

    for name, keywords in hierarchy:
        if any(k in t for k in keywords):
            return name

    return "unknown"


def authority_score(text: str):

    t = text.lower()

    score = 0.0


    # Strongest signal: explicit active and authoritative marker
    if "active and authoritative" in t:
        score += 10.0

    if "governing status: active" in t:
        score += 8.0

    # Supersession language
    if "supersedes all previous" in t:
        score += 4.0
    elif "supersedes earlier" in t:
        score += 4.0
    elif "supersedes previous" in t:
        score += 3.0

    # Standard authority markers
    if "active governing policy" in t:
        score += 5.0
    elif "governing policy" in t:
        score += 4.0

    if "current policy" in t:
        score += 3.0

    if "currently effective" in t:
        score += 3.0

    if "effective immediately" in t:
        score += 2.5

    if "official policy" in t:
        score += 2.5

    if "policy bulletin" in t:
        score += 2.0

    if "official standard" in t:
        score += 2.0

    # ------------------------
    # Negative authority
    # ------------------------

    # Explicit disclaimers (very negative)
    if "should not be treated as the governing source" in t:
        score -= 8.0

    if "consult the active governing policy if a conflict exists" in t:
        score -= 6.0

    if "does not state the current requirement" in t:
        score -= 7.0

    # Status markers
    if "superseded" in t and "supersedes" not in t:
        score -= 6.0

    if "archived" in t:
        score -= 5.0

    if "stale" in t:
        score -= 5.0

    if "draft" in t:
        score -= 4.0

    # Document type markers (secondary sources)
    if "secondary internal summary" in t:
        score -= 4.0

    if "implementation memo" in t:
        score -= 3.0

    if "training guide" in t:
        score -= 2.5

    if "planning brief" in t:
        score -= 2.5

    if "compliance digest" in t:
        score -= 2.0

    if "reference card" in t:
        score -= 2.0

    if "background note" in t:
        score -= 3.0

    if "faq" in t:
        score -= 1.5

    # Scope issues
    if "wrong scope" in t:
        score -= 5.0

    if "neighboring category" in t:
        score -= 4.0

    if "not the target category" in t:
        score -= 4.0

    return score


# ============================================================
# CLAIM EXTRACTION
# ============================================================

QUANTITY_PATTERN = re.compile(
    r"""
    (?:
        -?\d+(?:,\d{3})*(?:\.\d+)?
        \s*
        (?:
            %|
            °c|
            degrees?\s+celsius|
            minutes?|
            hours?|
            days?|
            years?|
            kilometers?|
            km|
            cases?|
            calls?|
            quens?|
            luma|
            inspection\s+points?
        )?
    )
    """,
    re.IGNORECASE | re.VERBOSE
)


def keyword_overlap(question: str, sentence: str):

    q_words = set(
        re.findall(r"[a-zA-Z]{3,}", question.lower())
    )

    s_words = set(
        re.findall(r"[a-zA-Z]{3,}", sentence.lower())
    )

    if not q_words:
        return 0

    return len(q_words & s_words) / len(q_words)


def extract_claim(question: str, text: str):

    """
    Find numerical / quantitative claims in sentences.
    Pick the claim whose sentence is most related to the question.
    """

    candidates = []

    for sent in sentences(text):

        matches = QUANTITY_PATTERN.findall(sent)

        if not matches:
            continue

        relevance = keyword_overlap(question, sent)

        for value in matches:
            value = value.strip()

            candidates.append(
                (
                    relevance,
                    value,
                    sent
                )
            )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    _, value, sentence = candidates[0]

    return value, sentence


# ============================================================
# CLAIM NORMALIZATION
# ============================================================

def normalize_claim(claim):

    if not claim:
        return "NO_CLAIM"

    x = claim.lower().strip()

    x = x.replace(",", "")
    x = x.replace("kilometers", "km")
    x = x.replace("kilometer", "km")
    x = x.replace("degrees celsius", "°c")

    x = re.sub(r"\s+", " ", x)

    return x


# ============================================================
# STAGE 1: BROAD SEMANTIC RETRIEVAL
# ============================================================

def semantic_retrieve(
    question,
    documents,
    embedder,
    k=INITIAL_RETRIEVAL_K
):

    q_embedding = embedder.encode([question])[0]

    scored = []

    for doc in documents:

        score = cosine_similarity(
            q_embedding,
            doc.embedding
        )

        scored.append(
            (score, doc)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored[:k], q_embedding


# ============================================================
# SCOPE RELEVANCE
# ============================================================

def get_scope_score(
    question,
    document_text,
    embedder,
    q_embedding
):

    # Use first several sentences because scope is often
    # described near title/header/intro.

    doc_sents = sentences(document_text)[:8]

    if not doc_sents:
        return 0.0

    embeddings = embedder.encode(doc_sents)

    scores = [
        cosine_similarity(q_embedding, e)
        for e in embeddings
    ]

    return max(scores)


# ============================================================
# NORMALIZE AUTHORITY SCORE
# ============================================================

def normalize_authority(raw):

    # expected approximate range [-10, +15]
    raw = max(-10, min(15, raw))

    return (raw + 10) / 25


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_candidates(
    question,
    retrieved,
    q_embedding,
    embedder
):

    candidates = []

    for semantic, doc in retrieved:

        raw_auth = authority_score(doc.text)

        auth = normalize_authority(raw_auth)

        scope = get_scope_score(
            question,
            doc.text,
            embedder,
            q_embedding
        )

        claim, claim_sentence = extract_claim(
            question,
            doc.text
        )

        candidate = Candidate(
            filename=doc.filename,

            text=doc.text,

            semantic_score=semantic,
            authority_score=auth,
            scope_score=scope,

            claim=claim,
            claim_sentence=claim_sentence,

            status=detect_status(doc.text),
            doc_type=detect_doc_type(doc.text)
        )

        candidate.final_score = (
            W_SEMANTIC * candidate.semantic_score
            +
            W_AUTHORITY * candidate.authority_score
            +
            W_SCOPE * candidate.scope_score
        )

        candidates.append(candidate)

    return candidates


# ============================================================
# STAGE 2: CONFLICT GROUPING
# ============================================================

def group_by_claim(candidates):

    groups = defaultdict(list)

    for candidate in candidates:

        key = normalize_claim(candidate.claim)

        groups[key].append(candidate)

    return groups


# ============================================================
# STAGE 3: CONFLICT-AWARE DIVERSIFICATION
# ============================================================

def diversify_conflicts(
    candidates,
    max_per_claim=1
):

    """
    Prevent five documents containing the same incorrect
    claim from occupying five context positions.
    """

    groups = group_by_claim(candidates)

    diversified = []

    for claim, docs in groups.items():

        docs.sort(
            key=lambda x: x.final_score,
            reverse=True
        )

        diversified.extend(
            docs[:max_per_claim]
        )

    diversified.sort(
        key=lambda x: x.final_score,
        reverse=True
    )

    return diversified


# ============================================================
# STAGE 4: AUTHORITY-FIRST RESOLUTION
# ============================================================

def authority_resolve(candidates):

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda c: c.final_score,
        reverse=True
    )

    return candidates[0]


# ============================================================
# EXPLAIN RESULTS
# ============================================================

def print_conflict_table(candidates):

    print("\n" + "=" * 100)

    print(
        f"{'FILE':30} "
        f"{'CLAIM':15} "
        f"{'STATUS':12} "
        f"{'TYPE':20} "
        f"{'SEM':7} "
        f"{'AUTH':7} "
        f"{'SCOPE':7} "
        f"{'FINAL':7}"
    )

    print("=" * 100)

    for c in candidates:

        print(
            f"{c.filename[:30]:30} "
            f"{str(c.claim)[:15]:15} "
            f"{c.status[:12]:12} "
            f"{c.doc_type[:20]:20} "
            f"{c.semantic_score:.3f}   "
            f"{c.authority_score:.3f}   "
            f"{c.scope_score:.3f}   "
            f"{c.final_score:.3f}"
        )


# ============================================================
# GOV-RAG
# ============================================================

class GovRAG:

    def __init__(self, doc_dir):

        self.embedder = Embedder()

        self.documents = load_documents(doc_dir)

        texts = [
            d.text
            for d in self.documents
        ]

        print("Creating document embeddings...")

        embeddings = self.embedder.encode(texts)

        for doc, embedding in zip(
            self.documents,
            embeddings
        ):
            doc.embedding = embedding

        print("Ready.")


    def query(self, question):

        # ----------------------------------------------------
        # 1. Broad semantic retrieval
        # ----------------------------------------------------

        retrieved, q_embedding = semantic_retrieve(
            question,
            self.documents,
            self.embedder,
            INITIAL_RETRIEVAL_K
        )

        # ----------------------------------------------------
        # 2. Authority + scope extraction
        # ----------------------------------------------------

        candidates = build_candidates(
            question,
            retrieved,
            q_embedding,
            self.embedder
        )

        # ----------------------------------------------------
        # 3. Conflict diversification
        # ----------------------------------------------------

        diversified = diversify_conflicts(
            candidates,
            max_per_claim=1
        )

        diversified = diversified[
            :FINAL_CONTEXT_K
        ]

        # ----------------------------------------------------
        # 4. Resolve
        # ----------------------------------------------------

        winner = authority_resolve(
            diversified
        )

        print_conflict_table(
            diversified
        )

        if winner is None:
            return {
                "answer": None,
                "source": None,
                "message": "No usable evidence retrieved."
            }

        return {
            "answer": winner.claim,
            "source": winner.filename,
            "supporting_sentence": winner.claim_sentence,
            "status": winner.status,
            "document_type": winner.doc_type,
            "score": winner.final_score,
            "candidates": [
                asdict(x)
                for x in diversified
            ]
        }


# ============================================================
# MAIN - Interactive Mode
# ============================================================

def interactive_mode(doc_dir):
    """Run GOV-RAG in interactive mode."""
    rag = GovRAG(doc_dir)

    while True:
        question = input("\nQuestion (or 'exit'): ").strip()

        if question.lower() == "exit":
            break

        result = rag.query(question)

        print("\nGOV-RAG RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        doc_dir = sys.argv[1]
    else:
        print("Usage: python gov_rag.py <document_directory>")
        print("Example: python gov_rag.py conflictbench_fictional_full/packs/easy")
        sys.exit(1)
    
    interactive_mode(doc_dir)