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
from typing import List, Optional

import numpy as np

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

SYSTEM_PROMPT = """You are the final evidence-resolution component of GOV-RAG, a retrieval system
designed for question answering over conflicting documents.

You must answer the user's question using ONLY the retrieved evidence provided to you.

The retrieved documents may intentionally disagree.

Your task is not to choose the most frequently repeated answer.
Your task is to identify the answer supported by the governing source that applies
to the exact scope of the question.

When resolving conflicts, consider:

1. Exact scope:
   Prefer documents that apply to the exact entity, category, tier, product,
   model, incident class, or policy scope asked about.

2. Authority:
   Prefer active governing policies, official standards, and authoritative
   primary documents over FAQs, summaries, training guides, reference cards,
   planning briefs, compliance digests, or other secondary materials.

3. Current status:
   Prefer active/current documents over archived, superseded, obsolete,
   historical, or draft documents.

4. Supersession:
   If one document explicitly supersedes or replaces another, prefer the
   superseding document.

5. Directness:
   Prefer a document that directly states the requirement over one that merely
   summarizes or refers to another source.

6. Evidence frequency is NOT authority:
   Multiple secondary documents repeating the same value do not outweigh one
   clearly authoritative governing document.

Do not use prior knowledge or world knowledge.

Do not assume that the first retrieved document is correct.

If the evidence conflicts, explicitly resolve the conflict using authority,
scope, and status.

If no retrieved document provides sufficient authoritative evidence, state that
the answer cannot be determined from the retrieved evidence.

Return JSON only in this format:

{
  "answer": "<concise answer - just the value, not a full sentence>",
  "selected_source": "<filename>",
  "reason": "<brief explanation of why this source governs>",
  "conflict_detected": true,
  "confidence": "high|medium|low"
}

IMPORTANT: For the "answer" field, provide ONLY the concise value (e.g., "7,430 Quens" or "26 hours" or "13.6%"), NOT a full sentence. Do not include phrases like "The answer is" or repeat the question."""

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
            parts.append(f"""Document {i}:
Filename: {c.filename}
Status: {c.status}
Type: {c.doc_type}
Authority Score: {c.authority_score:.3f}
Scope Score: {c.scope_score:.3f}

Content:
{c.text[:1500]}...

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


# Import all the detection and scoring functions from original
# (For brevity, using import - in practice you'd copy them here)
from gov_rag import (
    detect_status, detect_doc_type, authority_score,
    extract_claim, normalize_claim, normalize_authority,
    group_by_claim
)


# ============================================================
# RETRIEVAL (adapted for Gemini)
# ============================================================

def semantic_retrieve(question, documents, embedder, k=INITIAL_RETRIEVAL_K):
    """Semantic retrieval using Gemini embeddings."""
    
    # Encode query with RETRIEVAL_QUERY task type
    q_embedding = embedder.encode([question], task_type="RETRIEVAL_QUERY")[0]
    
    scored = []
    
    for doc in documents:
        score = cosine_similarity(q_embedding, doc.embedding)
        scored.append((score, doc))
    
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

def build_candidates(question, retrieved, q_embedding, embedder):
    candidates = []
    
    for semantic, doc in retrieved:
        raw_auth = authority_score(doc.text)
        auth = normalize_authority(raw_auth)
        scope = get_scope_score(question, doc.text, embedder, q_embedding)
        claim, claim_sentence = extract_claim(question, doc.text)
        
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
            W_SEMANTIC * candidate.semantic_score +
            W_AUTHORITY * candidate.authority_score +
            W_SCOPE * candidate.scope_score
        )
        
        candidates.append(candidate)
    
    return candidates


def diversify_conflicts(candidates, max_per_claim=1):
    """Prevent conflict crowding."""
    groups = group_by_claim(candidates)
    diversified = []
    
    for claim, docs in groups.items():
        docs.sort(key=lambda x: x.final_score, reverse=True)
        diversified.extend(docs[:max_per_claim])
    
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
        
        # Create embeddings for all documents
        print(f"Creating embeddings for {len(self.documents)} documents...")
        texts = [d.text for d in self.documents]
        
        embeddings = self.embedder.encode(texts, task_type="RETRIEVAL_DOCUMENT")
        
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding
        
        print("Ready.")
    
    def query(self, question):
        """Query the GOV-RAG system with optional LLM generation.
        
        Pipeline:
        1. Question
        2. Broad semantic retrieval (top 20-30)
        3. Claim extraction
        4. Conflict grouping
        5. Authority + scope scoring
        6. Conflict-crowding control
        7. Top governing candidates
        8. GEMINI 2.5 PRO (if use_llm=True)
        9. Final answer + governing source
        """
        
        # 1. Broad semantic retrieval
        retrieved, q_embedding = semantic_retrieve(
            question, self.documents, self.embedder, INITIAL_RETRIEVAL_K
        )
        
        # 2. Authority + scope extraction
        candidates = build_candidates(question, retrieved, q_embedding, self.embedder)
        
        # 3. Conflict diversification
        diversified = diversify_conflicts(candidates, max_per_claim=1)
        diversified = diversified[:FINAL_CONTEXT_K]
        
        print_conflict_table(diversified)
        
        if not diversified:
            return {
                "answer": None,
                "source": None,
                "message": "No usable evidence retrieved.",
                "method": "none"
            }
        
        # 4. Final resolution: LLM or rule-based
        if self.use_llm and self.generator:
            print("\n" + "="*80)
            print("FINAL RESOLUTION: Gemini 2.5 Pro")
            print("="*80)
            
            llm_result = self.generator.generate(question, diversified)
            
            return {
                "answer": llm_result.get("answer"),
                "source": llm_result.get("selected_source"),
                "reason": llm_result.get("reason"),
                "conflict_detected": llm_result.get("conflict_detected", False),
                "confidence": llm_result.get("confidence", "medium"),
                "method": "llm",
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
                    "method": "rule-based"
                }
            
            return {
                "answer": winner.claim,
                "source": winner.filename,
                "supporting_sentence": winner.claim_sentence,
                "status": winner.status,
                "document_type": winner.doc_type,
                "score": winner.final_score,
                "method": "rule-based",
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
