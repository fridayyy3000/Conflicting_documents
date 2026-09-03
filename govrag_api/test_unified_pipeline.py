"""Regression tests for the unified GOV-RAG pipeline (chunking + conflict detection).

Exercises the full stack (main.py -> gov_rag_gemini.py -> corpus_store.py) through
the public /demo/corpora/* API against a locally running server.

Run the API first:
    export GOOGLE_CLOUD_PROJECT=... GOVRAG_API_SECRET=... VERTEX_AI_REGION=...
    python main.py
Then:
    python test_unified_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import requests


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{(' - ' + detail) if detail and not condition else ''}")
    if not condition:
        global FAILURES
        FAILURES += 1


FAILURES = 0


def create_corpus(api_url: str, name: str) -> str:
    resp = requests.post(f"{api_url}/demo/corpora", json={"name": name}, timeout=30)
    resp.raise_for_status()
    return resp.json()["corpus_id"]


def upload_files(api_url: str, corpus_id: str, files: dict) -> dict:
    file_tuples = [("files", (name, content, "text/markdown")) for name, content in files.items()]
    resp = requests.post(f"{api_url}/demo/corpora/{corpus_id}/documents", files=file_tuples, timeout=120)
    resp.raise_for_status()
    return resp.json()


def ask(api_url: str, corpus_id: str, question: str) -> dict:
    resp = requests.post(
        f"{api_url}/demo/corpora/{corpus_id}/query",
        json={"question": question},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def cleanup(api_url: str, corpus_id: str) -> None:
    try:
        requests.delete(f"{api_url}/demo/corpora/{corpus_id}", timeout=30)
    except requests.RequestException:
        pass


def test_a_single_document(api_url: str) -> None:
    print("\n=== TEST A: single simple document ===")
    corpus_id = create_corpus(api_url, "test-a")
    try:
        upload_files(api_url, corpus_id, {
            "auth.md": "Authentication tokens expire after 30 minutes.",
        })
        result = ask(api_url, corpus_id, "How long do authentication tokens last?")
        print(f"answer={result.get('answer')!r} conflict_detected={result.get('conflict_detected')}")
        check("30 minutes" in (result.get("answer") or ""), "A: answer contains '30 minutes'", str(result.get("answer")))
        check(result.get("conflict_detected") is False, "A: conflict_detected is false")
    finally:
        cleanup(api_url, corpus_id)


def test_b_agreeing_documents(api_url: str) -> None:
    print("\n=== TEST B: several agreeing documents ===")
    corpus_id = create_corpus(api_url, "test-b")
    try:
        upload_files(api_url, corpus_id, {
            "doc1.md": "The authentication timeout is 30 minutes for all users.",
            "doc2.md": "Sessions authenticate and remain valid for 30 minutes before requiring re-login.",
            "doc3.md": "Per company policy, authentication expires after 30 minutes of inactivity.",
        })
        result = ask(api_url, corpus_id, "What is the authentication timeout?")
        print(f"answer={result.get('answer')!r} conflict_detected={result.get('conflict_detected')}")
        check("30 minutes" in (result.get("answer") or ""), "B: answer contains '30 minutes'", str(result.get("answer")))
        check(result.get("conflict_detected") is False, "B: conflict_detected is false")
    finally:
        cleanup(api_url, corpus_id)


def test_c_conflicting_documents(api_url: str) -> None:
    print("\n=== TEST C: conflicting documents ===")
    corpus_id = create_corpus(api_url, "test-c")
    try:
        upload_files(api_url, corpus_id, {
            "official.md": (
                "# Official Policy\n\nStatus: ACTIVE AND AUTHORITATIVE\n\n"
                "The authentication timeout is 30 minutes."
            ),
            "draft.md": (
                "# Draft Update\n\nStatus: DRAFT\n\n"
                "The proposed authentication timeout is 60 minutes."
            ),
        })
        result = ask(api_url, corpus_id, "What is the current authentication timeout?")
        print(f"answer={result.get('answer')!r} selected_source={result.get('selected_source')} "
              f"conflict_detected={result.get('conflict_detected')}")
        check("30 minutes" in (result.get("answer") or ""), "C: answer contains '30 minutes'", str(result.get("answer")))
        check(result.get("conflict_detected") is True, "C: conflict_detected is true")
    finally:
        cleanup(api_url, corpus_id)


def test_d_long_document_retrieval(api_url: str) -> None:
    print("\n=== TEST D: long-document retrieval (answer far from start) ===")
    corpus_id = create_corpus(api_url, "test-d")
    try:
        filler = " ".join(
            f"This paragraph number {i} discusses unrelated background context, "
            f"historical notes, and general system architecture considerations "
            f"that are not directly relevant to the specific question being tested."
            for i in range(220)
        )
        needle = (
            "In the final evaluation section, the measured average latency "
            "improvement was 37.5 percent over the previous baseline implementation."
        )
        long_doc = f"{filler}\n\n{needle}\n\n{filler}"
        word_count = len(long_doc.split())
        print(f"long_doc word count: {word_count} (needle placed roughly mid-document)")

        upload_files(api_url, corpus_id, {"long_report.md": long_doc})
        result = ask(api_url, corpus_id, "What was the average latency improvement over the baseline?")
        print(f"answer={result.get('answer')!r} conflict_detected={result.get('conflict_detected')}")
        check("37.5" in (result.get("answer") or ""), "D: answer contains '37.5'", str(result.get("answer")))
        check(result.get("conflict_detected") is False, "D: conflict_detected is false")
    finally:
        cleanup(api_url, corpus_id)


def test_e_research_paper(api_url: str) -> None:
    print("\n=== TEST E: research-paper style document ===")
    quiett_candidates = list(Path(".").rglob("*QUIETT*.pdf")) + list(Path("..").rglob("*QUIETT*.pdf"))
    if quiett_candidates:
        pdf_path = quiett_candidates[0]
        print(f"Found QUIETT PDF at {pdf_path}, testing against the real document.")
        corpus_id = create_corpus(api_url, "test-e")
        try:
            content = pdf_path.read_bytes()
            file_tuples = [("files", (pdf_path.name, content, "application/pdf"))]
            resp = requests.post(f"{api_url}/demo/corpora/{corpus_id}/documents", files=file_tuples, timeout=120)
            resp.raise_for_status()

            r1 = ask(api_url, corpus_id, "What is QUIETT based on?")
            print(f"Q1 answer={r1.get('answer')!r}")
            check(bool(r1.get("answer")) and "cannot be determined" not in (r1.get("answer") or "").lower(),
                  "E: Q1 produced a real answer")
        finally:
            cleanup(api_url, corpus_id)
    else:
        print("QUIETT PDF not found locally; skipping (allowed per spec).")

        # Synthetic stand-in proving retrieval reaches beyond the abstract of a
        # realistically long research-paper-style document, without hardcoding
        # any QUIETT-specific answer.
        corpus_id = create_corpus(api_url, "test-e-synthetic")
        try:
            filler = " ".join(
                f"Related work paragraph {i} reviews prior approaches, background "
                f"context, and motivation for the proposed method in this area of study."
                for i in range(250)
            )
            paper = f"""Abstract
SYNTHIA is a transform-first, query-later framework for efficient retrieval.

1. Introduction
{filler}

7. Results
SYNTHIA achieves an average gain of 12.3% over the best baseline across all benchmarks.
"""
            upload_files(api_url, corpus_id, {"synthetic_paper.md": paper})

            r1 = ask(api_url, corpus_id, "What is SYNTHIA based on?")
            print(f"Q1 (abstract) answer={r1.get('answer')!r}")
            check("transform" in (r1.get("answer") or "").lower(), "E: abstract question answered", str(r1.get("answer")))

            r2 = ask(api_url, corpus_id, "What is SYNTHIA's average gain over the best baseline?")
            print(f"Q2 (late section) answer={r2.get('answer')!r} status={[s['status'] for s in r2.get('top_sources', [])]} "
                  f"doc_type={[s['document_type'] for s in r2.get('top_sources', [])]}")
            check("12.3" in (r2.get("answer") or ""), "E: late-section answer retrieved", str(r2.get("answer")))
            check(
                all(s.get("status") == "unknown" for s in r2.get("top_sources", [])),
                "E: no fabricated governance status on a normal paper",
                str([s.get("status") for s in r2.get("top_sources", [])]),
            )
        finally:
            cleanup(api_url, corpus_id)


def test_conflictbench_regression(api_url: str, api_key: str | None) -> None:
    print("\n=== ConflictBench Easy regression (via /demo_query, unauthenticated) ===")
    resp = requests.post(
        f"{api_url}/demo_query",
        json={
            "question": "What is the maximum defect rate allowed for Q4-certified suppliers at Meridian Forge?",
            "use_llm": True,
            "top_k": 8,
        },
        timeout=600,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"answer={result.get('answer')!r} selected_source={result.get('selected_source')!r} "
          f"conflict_detected={result.get('conflict_detected')} confidence={result.get('confidence')}")

    check(result.get("answer") == "0.72%", "ConflictBench: answer == '0.72%'", str(result.get("answer")))
    check(result.get("selected_source") == "Q009_source_12.md", "ConflictBench: selected_source == 'Q009_source_12.md'",
          str(result.get("selected_source")))
    check(result.get("conflict_detected") is True, "ConflictBench: conflict_detected is true")
    check(result.get("confidence") == "high", "ConflictBench: confidence == 'high'", str(result.get("confidence")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    test_conflictbench_regression(args.api_url, os.getenv("GOVRAG_API_KEY"))
    test_a_single_document(args.api_url)
    test_b_agreeing_documents(args.api_url)
    test_c_conflicting_documents(args.api_url)
    test_d_long_document_retrieval(args.api_url)
    test_e_research_paper(args.api_url)

    print(f"\n{'='*60}\n{FAILURES} failure(s)\n{'='*60}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
