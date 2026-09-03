"""End-to-end dynamic corpus workflow test for GOV-RAG API."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import requests


def build_headers(api_key: str | None) -> dict:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def write_sample_docs(tmp_dir: Path) -> list[Path]:
    docs = {
        "official_policy.md": """# Official Policy Bulletin

Entity: Meridian Supplier Program
Scope: Q4-certified suppliers
Status: ACTIVE AND AUTHORITATIVE

The maximum defect rate allowed for Q4-certified suppliers is 0.72%.
This bulletin supersedes all previous summaries and drafts.
""",
        "draft_policy.md": """# Draft Update

Entity: Meridian Supplier Program
Scope: Q4-certified suppliers
Status: DRAFT

Proposed threshold in this draft is 0.90%.
""",
        "internal_summary.md": """# Internal Summary

Entity: Meridian Supplier Program
Scope: Q4-certified suppliers
Status: ACTIVE

This summary references an older value of 1.00%.
""",
    }

    paths = []
    for name, content in docs.items():
        path = tmp_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def check_response(resp: requests.Response, label: str) -> dict:
    if resp.status_code >= 400:
        raise RuntimeError(f"{label} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.getenv("GOVRAG_API_KEY"))
    parser.add_argument("--project-id", default="project-79920195-9e86-44ea-8c9")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--use-llm", action="store_true", default=False)
    parser.add_argument("--skip-demo-query", action="store_true", default=False)
    args = parser.parse_args()

    headers = build_headers(args.api_key)

    print("1) GET /health")
    health = check_response(requests.get(f"{args.api_url}/health", timeout=30), "health")
    print(json.dumps(health, indent=2))

    if not args.skip_demo_query:
        print("2) POST /demo_query")
        demo = check_response(
            requests.post(
                f"{args.api_url}/demo_query",
                json={
                    "question": "What is the maximum defect rate allowed for Q4-certified suppliers at Meridian Forge?",
                    "project_id": args.project_id,
                    "region": args.region,
                    "use_llm": args.use_llm,
                    "top_k": 8,
                },
                timeout=180,
            ),
            "demo_query",
        )
        print(f"demo answer={demo.get('answer')} source={demo.get('selected_source')}")

    print("3) POST /corpora")
    created = check_response(
        requests.post(
            f"{args.api_url}/corpora",
            headers=headers,
            json={"name": "Workflow Test Corpus"},
            timeout=30,
        ),
        "create_corpus",
    )
    corpus_id = created["corpus_id"]
    print(f"corpus_id={corpus_id}")

    with tempfile.TemporaryDirectory() as d:
        doc_paths = write_sample_docs(Path(d))

        print("4) POST /corpora/{id}/documents (3 files)")
        files = [("files", (p.name, p.read_bytes(), "text/markdown")) for p in doc_paths]
        uploaded = check_response(
            requests.post(
                f"{args.api_url}/corpora/{corpus_id}/documents",
                headers=headers,
                files=files,
                timeout=120,
            ),
            "upload_documents",
        )
        print(json.dumps(uploaded, indent=2))

        print("5) POST /corpora/{id}/index")
        indexed = check_response(
            requests.post(
                f"{args.api_url}/corpora/{corpus_id}/index",
                headers=headers,
                json={
                    "project_id": args.project_id,
                    "region": args.region,
                    "use_llm": args.use_llm,
                },
                timeout=300,
            ),
            "index_corpus",
        )
        print(json.dumps(indexed, indent=2))

        print("6) POST /corpora/{id}/query")
        q1 = check_response(
            requests.post(
                f"{args.api_url}/corpora/{corpus_id}/query",
                headers=headers,
                json={
                    "question": "What defect rate currently applies to Q4-certified suppliers?",
                    "project_id": args.project_id,
                    "region": args.region,
                    "use_llm": args.use_llm,
                    "top_k": 8,
                },
                timeout=300,
            ),
            "query_corpus_1",
        )
        print(f"answer={q1.get('answer')} selected_source={q1.get('selected_source')}")

        print("7) POST /corpora/{id}/documents (add one)")
        add_doc = Path(d) / "new_authoritative_policy.md"
        add_doc.write_text(
            """# New Authoritative Bulletin

Status: ACTIVE AND AUTHORITATIVE
Scope: Q4-certified suppliers

The updated maximum defect rate allowed is 0.68%.
""",
            encoding="utf-8",
        )
        uploaded2 = check_response(
            requests.post(
                f"{args.api_url}/corpora/{corpus_id}/documents",
                headers=headers,
                files=[("files", (add_doc.name, add_doc.read_bytes(), "text/markdown"))],
                timeout=120,
            ),
            "upload_documents_2",
        )
        print(json.dumps(uploaded2, indent=2))

        print("8) GET /corpora/{id} (expect needs_index)")
        corpus_state = check_response(
            requests.get(
                f"{args.api_url}/corpora/{corpus_id}",
                headers=headers,
                timeout=30,
            ),
            "get_corpus",
        )
        print(f"status={corpus_state.get('status')} document_count={corpus_state.get('document_count')}")

        print("9) POST /corpora/{id}/index")
        indexed2 = check_response(
            requests.post(
                f"{args.api_url}/corpora/{corpus_id}/index",
                headers=headers,
                json={
                    "project_id": args.project_id,
                    "region": args.region,
                    "use_llm": args.use_llm,
                },
                timeout=300,
            ),
            "index_corpus_2",
        )
        print(json.dumps(indexed2, indent=2))

        print("10) POST /corpora/{id}/query")
        q2 = check_response(
            requests.post(
                f"{args.api_url}/corpora/{corpus_id}/query",
                headers=headers,
                json={
                    "question": "What defect rate currently applies to Q4-certified suppliers?",
                    "project_id": args.project_id,
                    "region": args.region,
                    "use_llm": args.use_llm,
                    "top_k": 8,
                },
                timeout=300,
            ),
            "query_corpus_2",
        )
        print(f"answer={q2.get('answer')} selected_source={q2.get('selected_source')}")

    print("Workflow test completed.")


if __name__ == "__main__":
    main()
