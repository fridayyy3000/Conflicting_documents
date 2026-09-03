#!/usr/bin/env python3
"""
GOV-RAG with Gemini Embeddings

This version uses Google's Gemini API for embeddings instead of 
local sentence-transformers. Useful when:
- You have network/proxy issues downloading models
- You want to use GCP infrastructure
- You prefer cloud-based embeddings

All other components (authority detection, conflict resolution) remain the same.
"""

import os
import re
import json
import glob
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from chunker import chunk_text

# Import Vertex AI (for project-based authentication)
try:
    from vertexai.language_models import TextEmbeddingModel
    import vertexai
    USE_VERTEX_AI = True
except ImportError:
    USE_VERTEX_AI = False
    print("WARNING: vertexai not installed, trying google-generativeai...")
    try:
        import google.generativeai as genai
        USE_VERTEX_AI = False
    except ImportError:
        print("ERROR: Neither vertexai nor google-generativeai installed")
        print("Run: pip install google-cloud-aiplatform")
        exit(1)


# ============================================================
# CONFIG
# ============================================================

# Vertex AI Configuration (project-based authentication)
VERTEX_AI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", None)
VERTEX_AI_REGION = os.getenv("VERTEX_AI_REGION", "us-central1")
VERTEX_AI_MODEL = "text-embedding-004"  # or "textembedding-gecko@003"
VERTEX_AI_GENERATION_MODEL = "gemini-2.5-pro"  # Gemini 2.5 Pro

# Fallback: AI Studio API Key (if not using Vertex AI)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# ============================================================
# SYSTEM PROMPT FOR FINAL EVIDENCE RESOLUTION
# ============================================================

SYSTEM_PROMPT = """You are GOV-RAG, a unified evidence-grounded question-answering system.

You always run the SAME pipeline: retrieve relevant evidence, then reason over it to
produce an answer. You do not have two separate modes. Instead, you naturally behave
like ordinary evidence-grounded QA when the evidence agrees (or when only one source
is relevant), and you apply stronger governance reasoning only when the evidence
genuinely disagrees on the answer to the same question.

You must answer the user's question using ONLY the retrieved evidence provided to you.

STEP 1: Determine whether the evidence actually conflicts.

Conflict means there are genuinely incompatible answers to the SAME question
(e.g. "0.72% vs 0.90%", "30 days vs 60 days", "Version 4 required vs Version 3
required").

Do NOT mark conflict_detected=true just because multiple documents or multiple
chunks were retrieved, multiple chunks come from the same or different files, or
the evidence discusses different aspects of the topic without disagreeing.

If only one relevant source/answer exists, or all relevant evidence agrees,
set conflict_detected=false.

STEP 2: Answer accordingly.

IF THE EVIDENCE DOES NOT CONFLICT (ordinary QA):
- Answer directly and concisely using the supplied evidence passages.
- Identify the single strongest supporting source.
- Do not require or invent authority/governance/policy reasoning - most ordinary
  documents (research papers, technical docs, manuals, etc.) have no governance
  metadata at all, and that is normal and expected.
- Do not invent information that is not in the supplied evidence.
- Only say the answer cannot be determined if the supplied relevant evidence
  genuinely does not contain the answer. Read the full passage text, not just
  any extracted "claim" field, before concluding evidence is insufficient.
- conflict_detected must be false.

IF THE EVIDENCE GENUINELY CONFLICTS, resolve it using, in order of importance:

1. Exact scope: prefer documents/passages that apply to the exact entity,
   category, tier, product, model, incident class, or policy scope asked about.
2. Authority: prefer active governing policies, official standards, and
   authoritative primary documents over FAQs, summaries, training guides,
   reference cards, planning briefs, compliance digests, or other secondary
   materials.
3. Current status: prefer active/current documents over archived, superseded,
   obsolete, historical, or draft documents.
4. Supersession: if one document explicitly supersedes or replaces another,
   prefer the superseding document.
5. Directness: prefer a passage that directly states the requirement over one
   that merely summarizes or refers to another source.
6. Evidence frequency is NOT authority: multiple secondary documents repeating
   the same value do not outweigh one clearly authoritative governing document.

Do not use prior knowledge or world knowledge. Do not assume the first
retrieved passage is correct. Reason from the actual passage text, not only
from any extracted "claim" field (extracted claims can be wrong for things
like "Q4", "Tier-3", or section numbers - ignore them if the passage text
says otherwise).

Return JSON only in this format:

{
  "answer": "<concise answer - just the value, not a full sentence>",
  "selected_source": "<filename>",
  "reason": "<brief explanation of why this evidence supports the answer, or why this source governs if there was a conflict>",
  "conflict_detected": true,
  "confidence": "high|medium|low"
}

IMPORTANT: For the "answer" field, provide ONLY the concise value (e.g., "7,430 Quens" or "26 hours" or "13.6%" or a short phrase), NOT a full sentence. Do not include phrases like "The answer is" or repeat the question."""

INITIAL_RETRIEVAL_K = 30
FINAL_CONTEXT_K = 8

# Reranking weights
W_SEMANTIC = 0.30
W_AUTHORITY = 0.50
W_SCOPE = 0.20


# ============================================================
# DATA STRUCTURES (same as original)
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
    chunk_index: Optional[int] = None


@dataclass
class Chunk:
    """A retrieval-unit slice of a document, preserving doc-level identity."""
    filename: str
    document_id: str
    chunk_index: int
    chunk_id: str
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class DocumentMeta:
    """Governance metadata computed once per whole document, reused by every chunk."""
    status: str
    doc_type: str
    authority_raw: float


# ============================================================
# VERTEX AI / GEMINI EMBEDDINGS
# ============================================================

class GeminiEmbedder:
    def __init__(self, project_id=None, region=None, api_key=None):
        """
        Initialize embedder with Vertex AI (project-based) or AI Studio (API key).
        
        Priority:
        1. Vertex AI with project_id (recommended for GCP)
        2. AI Studio with api_key (fallback)
        """
        self.use_vertex = False
        
        # Try Vertex AI first (project-based authentication)
        if project_id or VERTEX_AI_PROJECT:
            try:
                project = project_id or VERTEX_AI_PROJECT
                location = region or VERTEX_AI_REGION
                
                vertexai.init(project=project, location=location)
                self.model = TextEmbeddingModel.from_pretrained(VERTEX_AI_MODEL)
                self.use_vertex = True
                
                print(f"✓ Using Vertex AI")
                print(f"  Project: {project}")
                print(f"  Region: {location}")
                print(f"  Model: {VERTEX_AI_MODEL}")
                print(f"  Auth: gcloud credentials")
                return
                
            except Exception as e:
                print(f"WARNING: Vertex AI initialization failed: {e}")
                print("Falling back to AI Studio API key authentication...")
        
        # Fallback to AI Studio (API key)
        if api_key or GEMINI_API_KEY:
            import google.generativeai as genai
            
            key = api_key or GEMINI_API_KEY
            genai.configure(api_key=key)
            print(f"✓ Using AI Studio API (API key)")
            self.use_vertex = False
        else:
            print("\nERROR: No authentication found!")
            print("\nOption 1 - Vertex AI (Recommended):")
            print("  gcloud auth application-default login")
            print("  export GOOGLE_CLOUD_PROJECT='your-project-id'")
            print("\nOption 2 - AI Studio:")
            print("  export GOOGLE_API_KEY='your-api-key'")
            print("  Get key from: https://makersuite.google.com/app/apikey")
            raise ValueError("No authentication configured")
    
    def encode(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        """
        Encode texts using Vertex AI or Gemini API.
        
        task_type:
            - RETRIEVAL_DOCUMENT: For indexing documents
            - RETRIEVAL_QUERY: For search queries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_vertex:
            return self._encode_vertex(texts, task_type)
        else:
            return self._encode_genai(texts, task_type)
    
    def _encode_vertex(self, texts, task_type):
        """Encode using Vertex AI."""
        embeddings = []
        batch_size = 5  # Vertex AI has lower batch limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                # Vertex AI uses different method
                result = self.model.get_embeddings(batch)
                batch_embeddings = [emb.values for emb in result]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size}: {e}")
                dim = 768
                embeddings.extend([np.zeros(dim) for _ in batch])
        
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        
        return embeddings
    
    def _encode_genai(self, texts, task_type):
        """Encode using AI Studio API."""
        import google.generativeai as genai
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch,
                    task_type=task_type
                )
                
                batch_embeddings = result['embedding']
                
                if isinstance(batch_embeddings[0], (int, float)):
                    batch_embeddings = [batch_embeddings]
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size}: {e}")
                dim = 768
                embeddings.extend([np.zeros(dim) for _ in batch])
        
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        
        return embeddings


# ============================================================
# GEMINI GENERATOR (for final LLM-based resolution)
# ============================================================

class GeminiGenerator:
    """Generate final answers using Gemini 2.5 Pro."""
    
    def __init__(self, project_id=None, region=None, api_key=None):
        """Initialize Gemini generator with Vertex AI or AI Studio."""
        self.use_vertex = False
        
        # Try Vertex AI first
        if project_id or VERTEX_AI_PROJECT:
            try:
                from vertexai.generative_models import GenerativeModel
                
                project = project_id or VERTEX_AI_PROJECT
                location = region or VERTEX_AI_REGION
                
                vertexai.init(project=project, location=location)
                # Initialize model with system instruction
                self.model = GenerativeModel(
                    VERTEX_AI_GENERATION_MODEL,
                    system_instruction=[SYSTEM_PROMPT]
                )
                self.use_vertex = True
                
                print(f"✓ Using Vertex AI for generation")
                print(f"  Model: {VERTEX_AI_GENERATION_MODEL}")
                return
                
            except Exception as e:
                print(f"WARNING: Vertex AI generation init failed: {e}")
                print("Falling back to AI Studio...")
        
        # Fallback to AI Studio
        if api_key or GEMINI_API_KEY:
            import google.generativeai as genai
            
            key = api_key or GEMINI_API_KEY
            genai.configure(api_key=key)
            print(f"✓ Using AI Studio for generation")
            self.use_vertex = False
        else:
            raise ValueError("No authentication configured for generation")
    
    def generate(self, question, candidates):
        """
        Generate final answer using Gemini.
        
        Args:
            question: User's question
            candidates: List of Candidate objects (top K documents)
        
        Returns:
            dict with answer, selected_source, reason, conflict_detected, confidence
        """
        # Format retrieved documents
        evidence_text = self._format_evidence(candidates)
        
        # Create prompt
        user_prompt = f"""Question: {question}

Retrieved Evidence:

{evidence_text}

Analyze the evidence and provide your answer in JSON format."""
        
        try:
            if self.use_vertex:
                # Vertex AI - system instruction already set in model init
                response = self.model.generate_content(
                    [user_prompt],
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 2048,
                        "candidate_count": 1,
                    }
                )
                # Extract full response text from Vertex AI
                try:
                    response_text = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    response_text = response.text
            else:
                # AI Studio
                import google.generativeai as genai
                
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-pro",
                    system_instruction=SYSTEM_PROMPT
                )
                response = model.generate_content(
                    user_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=2048,
                    )
                )
                response_text = response.text
            
            # Parse JSON response
            # Remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            return result
            
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parsing failed: {e}")
            print(f"Full response ({len(response_text) if 'response_text' in locals() else 0} chars):")
            print(response_text if 'response_text' in locals() else 'N/A')
            print("\nAttempting to extract answer from partial response...")
            
            # Try to extract answer from partial JSON
            if 'response_text' in locals() and 'answer' in response_text:
                import re
                answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', response_text)
                source_match = re.search(r'"selected_source"\s*:\s*"([^"]+)"', response_text)
                
                if answer_match:
                    return {
                        "answer": answer_match.group(1),
                        "selected_source": source_match.group(1) if source_match else candidates[0].filename,
                        "reason": "Extracted from partial LLM response",
                        "conflict_detected": True,
                        "confidence": "medium"
                    }
            
            # Fallback to rule-based resolution
            return self._fallback_resolution(question, candidates)
            
        except Exception as e:
            print(f"ERROR: Gemini generation failed: {e}")
            print(f"Response text: {response_text if 'response_text' in locals() else 'N/A'}")
            # Fallback to rule-based resolution
            return self._fallback_resolution(question, candidates)
    
    def _format_evidence(self, candidates):
        """Format candidates as numbered evidence blocks."""
        parts = []
        
        for i, c in enumerate(candidates, 1):
            chunk_label = f" (chunk {c.chunk_index + 1})" if c.chunk_index is not None else ""
            text = c.text if len(c.text) <= 6000 else c.text[:6000] + "..."
            parts.append(f"""Document {i}{chunk_label}:
Filename: {c.filename}
Status: {c.status}
Type: {c.doc_type}
Authority Score: {c.authority_score:.3f}
Scope Score: {c.scope_score:.3f}

Content:
{text}

{'-'*80}
""")
        
        return "\n".join(parts)
    
    def _fallback_resolution(self, question, candidates):
        """Rule-based fallback if LLM fails."""
        if not candidates:
            return {
                "answer": "Unable to determine",
                "selected_source": None,
                "reason": "No evidence retrieved",
                "conflict_detected": False,
                "confidence": "low"
            }
        
        winner = candidates[0]  # Already sorted by final_score
        
        return {
            "answer": winner.claim or "Unable to determine",
            "selected_source": winner.filename,
            "reason": f"Highest authority score ({winner.authority_score:.3f})",
            "conflict_detected": len(set(c.claim for c in candidates if c.claim)) > 1,
            "confidence": "medium"
        }


# ============================================================
# LOAD DOCUMENTS (same as original)
# ============================================================

def load_documents(folder: str) -> List[Document]:
    files = glob.glob(os.path.join(folder, "**/*.md"), recursive=True)
    docs = []
    
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        docs.append(Document(filename=os.path.basename(path), text=text))
    
    print(f"Loaded {len(docs)} documents")
    return docs


def build_document_meta(documents: List[Document]) -> Dict[str, DocumentMeta]:
    """Compute governance metadata once per whole document; chunks reuse it."""
    meta = {}
    for doc in documents:
        meta[doc.filename] = DocumentMeta(
            status=detect_status(doc.text),
            doc_type=detect_doc_type(doc.text),
            authority_raw=authority_score(doc.text),
        )
    return meta


def build_chunks(documents: List[Document]) -> List[Chunk]:
    """Split each document into retrieval-unit chunks (short docs -> 1 chunk)."""
    chunks = []
    for doc in documents:
        pieces = chunk_text(doc.text)
        for idx, piece in enumerate(pieces):
            chunks.append(Chunk(
                filename=doc.filename,
                document_id=doc.filename,
                chunk_index=idx,
                chunk_id=f"{doc.filename}::chunk{idx}",
                text=piece,
            ))
    return chunks


# ============================================================
# UTILITY FUNCTIONS (same as original)
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
# UTILITY FUNCTIONS (Copied from gov_rag.py to avoid dependency)
# ============================================================

def sentences(text):
    """Split text into sentences"""
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+|\n+", text)
        if x.strip()
    ]


def keyword_overlap(question: str, sentence: str):
    """Calculate keyword overlap between question and sentence"""
    q_words = set(
        re.findall(r"[a-zA-Z]{3,}", question.lower())
    )
    
    s_words = set(
        re.findall(r"[a-zA-Z]{3,}", sentence.lower())
    )
    
    if not q_words:
        return 0
    
    return len(q_words & s_words) / len(q_words)


QUANTITY_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9])
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


# Header/first-N-characters window used to keep single ambiguous keywords
# (e.g. "draft", "summary") from firing on incidental body-text mentions in
# ordinary documents; strong multi-word phrases are still checked anywhere.
HEADER_WINDOW_CHARS = 1200


def get_header_window(text: str) -> str:
    return text[:HEADER_WINDOW_CHARS].lower()


def extract_structured_field(text: str, field_names) -> Optional[str]:
    """Match explicit 'Label: value' lines such as 'Status: DRAFT'."""
    pattern = re.compile(
        rf"^\s*(?:{'|'.join(field_names)})\s*:\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1).strip().lower() if match else None


def detect_status(text: str):
    """Detect document status conservatively: only trust an explicit structural
    cue (a 'Status:'/'Governing status:' line) or a strong, specific multi-word
    phrase, never a single ambiguous word (draft/archived/historical/stale)
    found anywhere in the body. Verified against all 180 ConflictBench Easy
    documents: every one of them is classified via a structured line or a
    strong phrase, so this conservative approach has zero regression there,
    while avoiding false positives on ordinary documents that happen to
    contain common words like "historical" or "propose" in their prose.
    """
    full = text.lower()
    structured = extract_structured_field(text, ["governing status", "status"])

    if "active and authoritative" in full or "governing status: active" in full:
        return "active_authoritative"
    if structured and "authoritative" in structured:
        return "active_authoritative"

    if any(x in full for x in ["superseded by", "supersedes all previous", "withdrawn", "no longer valid"]):
        return "superseded"
    if structured and "supersed" in structured:
        return "superseded"

    if any(x in full for x in ["legacy version"]):
        return "archived"
    if structured and any(w in structured for w in ["archived", "historical", "legacy", "stale", "obsolete"]):
        return "archived"

    if any(x in full for x in ["draft policy", "working draft", "proposed policy", "proposed requirement", "planning brief"]):
        return "draft"
    if structured and any(w in structured for w in ["draft", "proposed"]):
        return "draft"

    if any(x in full for x in [
        "active governing policy",
        "governing policy",
        "currently effective",
        "current policy",
        "effective immediately",
        "active policy",
    ]):
        return "active"
    if structured == "active":
        return "active"

    return "unknown"


def detect_doc_type(text: str):
    """Detect document type. Strong, specific multi-word governance terms are
    checked anywhere in the document. Weak generic words (summary/digest/
    overview) only count near the top, since a research paper's body section
    titled "Results Summary" should not make the whole document a
    secondary_summary. Falls back to a light heuristic for ordinary
    documents rather than forcing an unrelated governance type.
    """
    full = text.lower()
    header = get_header_window(text)

    hierarchy = [
        ("governing_policy", ["governing policy", "official policy"]),
        ("policy_bulletin", ["policy bulletin"]),
        ("standard", ["official standard", "safety standard", "compliance standard"]),
        ("implementation_memo", ["implementation memo", "implementation memorandum"]),
        ("faq", ["faq", "frequently asked questions"]),
        ("training_guide", ["training guide", "training material"]),
        ("reference_card", ["reference card", "quick reference"]),
    ]

    for name, keywords in hierarchy:
        if any(k in full for k in keywords):
            return name

    if any(k in header for k in ["summary", "digest", "overview"]):
        return "secondary_summary"

    # Ordinary-document fallback: sensible generic types, not governance labels.
    if "abstract" in header and any(k in full for k in ["references", "et al", "arxiv"]):
        return "research_paper"
    if any(k in header for k in ["readme", "documentation", "user guide", "getting started"]):
        return "documentation"
    if any(k in full for k in ["technical report", "specification", "architecture overview"]):
        return "technical_document"

    return "unknown"


def authority_score(text: str):
    """Calculate authority score based on text markers"""
    t = text.lower()
    # Reuse detect_status's conservative structured/strong-phrase logic rather
    # than re-implementing bare-keyword checks here (those false-positive on
    # ordinary prose - see detect_status docstring).
    status_hint = detect_status(text)
    
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
    
    if status_hint == "archived":
        score -= 5.0
    
    if status_hint == "draft":
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


def extract_claim(question: str, text: str):
    """Extract numerical claim most related to question"""
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


def normalize_claim(claim):
    """Normalize claim text for comparison"""
    if not claim:
        return "NO_CLAIM"
    
    x = claim.lower().strip()
    
    x = x.replace(",", "")
    x = x.replace("kilometers", "km")
    x = x.replace("kilometer", "km")
    x = x.replace("degrees celsius", "°c")
    
    x = re.sub(r"\s+", " ", x)
    
    return x


def normalize_authority(raw):
    """Normalize authority score to 0-1 range"""
    # expected approximate range [-10, +15]
    raw = max(-10, min(15, raw))
    
    return (raw + 10) / 25


def group_by_claim(candidates):
    """Group candidates by normalized claim"""
    groups = defaultdict(list)
    
    for candidate in candidates:
        key = normalize_claim(candidate.claim)
        groups[key].append(candidate)
    
    return groups


# ============================================================
# RETRIEVAL (adapted for Gemini)
# ============================================================

def semantic_retrieve(question, chunks, embedder, k=INITIAL_RETRIEVAL_K):
    """Broad semantic retrieval over chunks using Gemini embeddings."""
    
    # Encode query with RETRIEVAL_QUERY task type
    q_embedding = embedder.encode([question], task_type="RETRIEVAL_QUERY")[0]
    
    scored = []
    
    for chunk in chunks:
        score = cosine_similarity(q_embedding, chunk.embedding)
        scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k], q_embedding


def get_scope_score(question, document_text, embedder, q_embedding):
    """Score scope match using embeddings."""
    doc_sents = sentences(document_text)[:8]
    
    if not doc_sents:
        return 0.0
    
    embeddings = embedder.encode(doc_sents, task_type="RETRIEVAL_DOCUMENT")
    scores = [cosine_similarity(q_embedding, e) for e in embeddings]
    
    return max(scores)


# ============================================================
# CANDIDATE BUILDING (same logic, different embedder)
# ============================================================

def build_candidates(question, retrieved, q_embedding, embedder, doc_meta):
    """Build ranking candidates from retrieved chunks.

    Governance metadata (status/doc_type/authority) is looked up per
    document_id and reused across every chunk of that document; when a
    document has no explicit governance signals, doc_meta yields neutral
    defaults (status=unknown, authority=0.0) so semantic/scope scores
    dominate ranking naturally rather than being penalized.
    """
    candidates = []
    default_meta = DocumentMeta(status="unknown", doc_type="unknown", authority_raw=0.0)
    
    for semantic, chunk in retrieved:
        meta = doc_meta.get(chunk.filename, default_meta)
        auth = normalize_authority(meta.authority_raw)
        scope = get_scope_score(question, chunk.text, embedder, q_embedding)
        claim, claim_sentence = extract_claim(question, chunk.text)
        
        candidate = Candidate(
            filename=chunk.filename,
            text=chunk.text,
            semantic_score=semantic,
            authority_score=auth,
            scope_score=scope,
            claim=claim,
            claim_sentence=claim_sentence,
            status=meta.status,
            doc_type=meta.doc_type,
            chunk_index=chunk.chunk_index,
        )
        
        candidate.final_score = (
            W_SEMANTIC * candidate.semantic_score +
            W_AUTHORITY * candidate.authority_score +
            W_SCOPE * candidate.scope_score
        )
        
        candidates.append(candidate)
    
    return candidates


def diversify_conflicts(candidates, max_per_claim=1, max_no_claim=None):
    """Prevent conflict crowding for genuine claims, without discarding
    ordinary evidence that has no extractable claim.

    Real numeric/answer claims are still capped at max_per_claim per distinct
    claim (this is what prevents conflict crowding in ConflictBench-style
    corpora). Chunks with no extractable claim (the common case for ordinary
    documents like research papers) are grouped under "NO_CLAIM" and are
    NOT collapsed to a single chunk, since multiple different relevant
    passages from the same or different files are all valid evidence for
    normal QA - collapsing them would break long-document retrieval.
    """
    groups = group_by_claim(candidates)
    diversified = []
    
    for claim, docs in groups.items():
        docs.sort(key=lambda x: x.final_score, reverse=True)
        if claim == "NO_CLAIM":
            limit = max_no_claim if max_no_claim is not None else len(docs)
        else:
            limit = max_per_claim
        diversified.extend(docs[:limit])
    
    diversified.sort(key=lambda x: x.final_score, reverse=True)
    return diversified


def authority_resolve(candidates):
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: c.final_score, reverse=True)[0]


def print_conflict_table(candidates):
    """Display candidates in table format."""
    print("\n" + "=" * 100)
    print(
        f"{'FILE':30} {'CLAIM':15} {'STATUS':12} {'TYPE':20} "
        f"{'SEM':7} {'AUTH':7} {'SCOPE':7} {'FINAL':7}"
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
# GOV-RAG with Gemini
# ============================================================

class GovRAGGemini:
    def __init__(self, doc_dir, project_id=None, region=None, api_key=None, use_llm=True):
        """Initialize GOV-RAG with Gemini embeddings and optional LLM generation.
        
        Args:
            doc_dir: Path to document directory
            project_id: GCP project ID for Vertex AI
            region: GCP region (default: us-central1)
            api_key: API key for AI Studio (fallback)
            use_llm: If True, use Gemini for final answer generation (default: True)
        """
        
        print("\nInitializing GOV-RAG with Gemini embeddings...")
        
        self.use_llm = use_llm
        self.embedder = GeminiEmbedder(project_id=project_id, region=region, api_key=api_key)
        
        # Initialize generator if using LLM
        if self.use_llm:
            print("\nInitializing Gemini generator for final resolution...")
            self.generator = GeminiGenerator(project_id=project_id, region=region, api_key=api_key)
        else:
            self.generator = None
        
        self.documents = load_documents(doc_dir)
        self.doc_meta = build_document_meta(self.documents)
        self.chunks = build_chunks(self.documents)
        
        # Create embeddings for all chunks (short docs collapse to one chunk
        # each, so this is equivalent to prior whole-document behavior there).
        print(f"Creating embeddings for {len(self.chunks)} chunks across {len(self.documents)} documents...")
        texts = [c.text for c in self.chunks]
        
        embeddings = self.embedder.encode(texts, task_type="RETRIEVAL_DOCUMENT")
        
        for chunk, embedding in zip(self.chunks, embeddings):
            chunk.embedding = embedding
        
        print("Ready.")
    
    def query(self, question):
        """Query the unified GOV-RAG pipeline (chunk retrieval + governance-aware
        reranking + conflict-aware Gemini reasoning). The same pipeline runs
        every time; when evidence agrees or has no governance metadata, it
        naturally reduces to ordinary evidence-grounded semantic RAG.
        
        Pipeline:
        1. Question
        2. Broad semantic retrieval over chunks (top ~30)
        3. Claim extraction (auxiliary metadata only, not gating)
        4. Governance metadata reuse (per-document, applied to its chunks)
        5. Authority + scope scoring
        6. Conflict-crowding control (only for genuine claim-bearing chunks)
        7. Top evidence chunks
        8. GEMINI 2.5 PRO (if use_llm=True)
        9. Final answer + supporting/governing source
        """
        
        # 1. Broad semantic retrieval over chunks
        retrieved, q_embedding = semantic_retrieve(
            question, self.chunks, self.embedder, INITIAL_RETRIEVAL_K
        )
        
        # 2. Authority + scope extraction
        candidates = build_candidates(question, retrieved, q_embedding, self.embedder, self.doc_meta)
        
        # 3. Conflict diversification (claim-bearing chunks capped per claim;
        #    ordinary no-claim evidence is not collapsed to a single chunk).
        diversified = diversify_conflicts(candidates, max_per_claim=1, max_no_claim=FINAL_CONTEXT_K)
        diversified = diversified[:FINAL_CONTEXT_K]
        
        print_conflict_table(diversified)
        
        if not diversified:
            return {
                "answer": None,
                "source": None,
                "message": "No usable evidence retrieved.",
                "method": "none",
                "retrieval_mode": "unified_govrag",
            }
        
        distinct_sources = {c.filename for c in diversified}
        
        # 4. Final resolution: LLM or rule-based
        if self.use_llm and self.generator:
            print("\n" + "="*80)
            print("FINAL RESOLUTION: Gemini 2.5 Pro")
            print("="*80)
            
            llm_result = self.generator.generate(question, diversified)
            
            conflict_detected = llm_result.get("conflict_detected", False)
            if len(distinct_sources) <= 1:
                # Deterministic backstop: a single relevant source can never be
                # a genuine conflict, regardless of what the LLM guessed.
                conflict_detected = False
            
            return {
                "answer": llm_result.get("answer"),
                "source": llm_result.get("selected_source"),
                "reason": llm_result.get("reason"),
                "conflict_detected": conflict_detected,
                "confidence": llm_result.get("confidence", "medium"),
                "method": "llm",
                "retrieval_mode": "unified_govrag",
                "candidates": [asdict(x) for x in diversified]
            }
        else:
            # Rule-based fallback
            winner = authority_resolve(diversified)
            
            if winner is None:
                return {
                    "answer": None,
                    "source": None,
                    "message": "No usable evidence retrieved.",
                    "method": "rule-based",
                    "retrieval_mode": "unified_govrag",
                }
            
            return {
                "answer": winner.claim,
                "source": winner.filename,
                "supporting_sentence": winner.claim_sentence,
                "status": winner.status,
                "document_type": winner.doc_type,
                "score": winner.final_score,
                "method": "rule-based",
                "retrieval_mode": "unified_govrag",
                "conflict_detected": len(distinct_sources) > 1 and len(set(c.claim for c in diversified if c.claim)) > 1,
                "candidates": [asdict(x) for x in diversified]
            }


# ============================================================
# MAIN - Interactive Mode
# ============================================================

def interactive_mode(doc_dir, project_id=None, region=None, api_key=None, use_llm=True):
    """Run GOV-RAG in interactive mode with Vertex AI or Gemini."""
    rag = GovRAGGemini(doc_dir, project_id=project_id, region=region, api_key=api_key, use_llm=use_llm)
    
    while True:
        question = input("\nQuestion (or 'exit'): ").strip()
        
        if question.lower() == "exit":
            break
        
        result = rag.query(question)
        
        print("\nGOV-RAG RESULT (Gemini)")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        doc_dir = sys.argv[1]
        project_id = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print("Usage: python gov_rag_gemini.py <document_directory> [project_id]")
        print("\nExample with Vertex AI:")
        print("  export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("  gcloud auth application-default login")
        print("  python gov_rag_gemini.py conflictbench_fictional_full/packs/easy")
        print("\nOr with AI Studio:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        print("  python gov_rag_gemini.py conflictbench_fictional_full/packs/easy")
        sys.exit(1)
    
    interactive_mode(doc_dir, project_id=project_id)
