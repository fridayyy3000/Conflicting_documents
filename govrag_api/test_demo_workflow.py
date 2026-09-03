"""End-to-end test of the public /demo/corpora browser API (no API key required)."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import requests


def check_response(resp: requests.Response, label: str, expect: int = 200) -> dict:
    if resp.status_code != expect:
        raise RuntimeError(f"{label} expected {expect}, got {resp.status_code}: {resp.text}")
    return resp.json() if resp.text else {}


def write_sample_docs(tmp_dir: Path) -> list[Path]:
    docs = {
        "official_policy.md": """# Official Policy Bulletin

Status: ACTIVE AND AUTHORITATIVE
Scope: Q4-certified suppliers

The maximum defect rate allowed for Q4-certified suppliers is 0.72%.
""",
        "draft_policy.md": """# Draft Update

Status: DRAFT
Scope: Q4-certified suppliers

Proposed threshold in this draft is 0.90%.
""",
    }
    paths = []
    for name, content in docs.items():
        path = tmp_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    print("1) POST /demo/corpora (no auth)")
    created = check_response(
        requests.post(f"{args.api_url}/demo/corpora", json={"name": "Browser Session"}, timeout=30),
        "create_demo_corpus",
    )
    corpus_id = created["corpus_id"]
    assert "expires_at" in created
    print(f"corpus_id={corpus_id} expires_at={created['expires_at']}")

    with tempfile.TemporaryDirectory() as d:
        doc_paths = write_sample_docs(Path(d))

        print("2) POST /demo/corpora/{id}/documents")
        files = [("files", (p.name, p.read_bytes(), "text/markdown")) for p in doc_paths]
        uploaded = check_response(
            requests.post(f"{args.api_url}/demo/corpora/{corpus_id}/documents", files=files, timeout=120),
            "upload_demo_documents",
        )
        print(json.dumps(uploaded, indent=2))

        print("3) POST /demo/corpora/{id}/query (auto-index, no /index call)")
        answered = check_response(
            requests.post(
                f"{args.api_url}/demo/corpora/{corpus_id}/query",
                json={"question": "What defect rate currently applies to Q4-certified suppliers?"},
                timeout=300,
            ),
            "query_demo_corpus",
        )
        print(f"answer={answered.get('answer')} selected_source={answered.get('selected_source')}")

        print("4) GET /demo/corpora/{id}/documents")
        docs = check_response(
            requests.get(f"{args.api_url}/demo/corpora/{corpus_id}/documents", timeout=30),
            "list_demo_documents",
        )
        print(json.dumps(docs, indent=2))

    print("5) Upload more files, still under the 500-file cap")
    with tempfile.TemporaryDirectory() as d:
        many_files = []
        for i in range(9):
            p = Path(d) / f"extra_{i}.md"
            p.write_text(f"# doc {i}\ncontent", encoding="utf-8")
            many_files.append(("files", (p.name, p.read_bytes(), "text/markdown")))
        resp = requests.post(f"{args.api_url}/demo/corpora/{corpus_id}/documents", files=many_files, timeout=60)
        print(f"9 more files (11 total, under 500 cap) -> HTTP {resp.status_code}")

    print("6) Reject unsupported extension")
    resp = requests.post(
        f"{args.api_url}/demo/corpora/{corpus_id}/documents",
        files=[("files", ("bad.exe", b"hello", "application/octet-stream"))],
        timeout=30,
    )
    print(f"exe upload -> HTTP {resp.status_code}")

    print("7) Question too long -> expect 422")
    resp = requests.post(
        f"{args.api_url}/demo/corpora/{corpus_id}/query",
        json={"question": "x" * 3000},
        timeout=30,
    )
    print(f"long question -> HTTP {resp.status_code}")

    print("8) GET a nonexistent demo corpus -> expect 404")
    resp = requests.get(f"{args.api_url}/demo/corpora/00000000-0000-0000-0000-000000000000", timeout=30)
    print(f"missing corpus -> HTTP {resp.status_code}")

    print("9) Confirm no public listing endpoint for demo corpora")
    resp = requests.get(f"{args.api_url}/demo/corpora", timeout=30)
    print(f"GET /demo/corpora -> HTTP {resp.status_code} (expected 404/405, not 200 with a list)")

    print("10) DELETE /demo/corpora/{id}")
    deleted = check_response(
        requests.delete(f"{args.api_url}/demo/corpora/{corpus_id}", timeout=30),
        "delete_demo_corpus",
    )
    print(json.dumps(deleted, indent=2))

    print("11) GET deleted corpus -> expect 404")
    resp = requests.get(f"{args.api_url}/demo/corpora/{corpus_id}", timeout=30)
    print(f"deleted corpus -> HTTP {resp.status_code}")

    print("Demo workflow test completed.")


if __name__ == "__main__":
    main()
