"""Persistence layer for corpora in GCS (or local fallback)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from google.cloud import storage


class CorpusStoreError(Exception):
    """Base error for corpus store operations."""


class CorpusNotFoundError(CorpusStoreError):
    """Raised when a corpus does not exist."""


class DocumentNotFoundError(CorpusStoreError):
    """Raised when a document does not exist."""


class CorpusStore:
    def __init__(self, base_dir: str, bucket_name: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.bucket_name = (bucket_name or "").strip() or None
        self.use_gcs = self.bucket_name is not None

        # Must match the "corpora/..." prefix used by _read_json/_write_json below.
        self.local_root = self.base_dir / "corpora"
        self.local_root.mkdir(parents=True, exist_ok=True)

        self.client = None
        self.bucket = None
        if self.use_gcs:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _corpus_prefix(corpus_id: str) -> str:
        return f"corpora/{corpus_id}"

    @staticmethod
    def _metadata_path(corpus_id: str) -> str:
        return f"corpora/{corpus_id}/metadata.json"

    @staticmethod
    def _doc_blob_path(corpus_id: str, normalized_filename: str) -> str:
        return f"corpora/{corpus_id}/documents/{normalized_filename}"

    def _local_corpus_dir(self, corpus_id: str) -> Path:
        return self.local_root / corpus_id

    def _local_metadata_file(self, corpus_id: str) -> Path:
        return self._local_corpus_dir(corpus_id) / "metadata.json"

    def _read_json(self, path: str) -> Dict:
        if self.use_gcs:
            blob = self.bucket.blob(path)
            if not blob.exists():
                raise CorpusNotFoundError("Corpus metadata not found")
            return json.loads(blob.download_as_text())

        file_path = self.base_dir / path
        if not file_path.exists():
            raise CorpusNotFoundError("Corpus metadata not found")
        return json.loads(file_path.read_text(encoding="utf-8"))

    def _write_json(self, path: str, data: Dict) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=True)

        if self.use_gcs:
            blob = self.bucket.blob(path)
            blob.upload_from_string(payload, content_type="application/json")
            return

        file_path = self.base_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(payload, encoding="utf-8")

    def _upload_text(self, path: str, text: str) -> None:
        if self.use_gcs:
            blob = self.bucket.blob(path)
            blob.upload_from_string(text, content_type="text/markdown")
            return

        file_path = self.base_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")

    def _download_text(self, path: str) -> str:
        if self.use_gcs:
            blob = self.bucket.blob(path)
            if not blob.exists():
                raise DocumentNotFoundError(f"Document content missing: {path}")
            return blob.download_as_text()

        file_path = self.base_dir / path
        if not file_path.exists():
            raise DocumentNotFoundError(f"Document content missing: {path}")
        return file_path.read_text(encoding="utf-8")

    def _delete_path(self, path: str) -> None:
        if self.use_gcs:
            self.bucket.blob(path).delete(if_generation_match=None)
            return

        file_path = self.base_dir / path
        if file_path.exists():
            file_path.unlink()

    def create_corpus(
        self,
        corpus_id: str,
        name: str,
        kind: str = "admin",
        ttl_hours: Optional[float] = None,
    ) -> Dict:
        if not name.strip():
            raise CorpusStoreError("Corpus name cannot be empty")

        metadata_path = self._metadata_path(corpus_id)
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(hours=ttl_hours)).isoformat() if ttl_hours else None

        metadata = {
            "corpus_id": corpus_id,
            "name": name.strip(),
            "kind": kind,
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
            "status": "created",
            "document_count": 0,
            "documents": [],
        }
        self._write_json(metadata_path, metadata)
        return metadata

    @staticmethod
    def is_expired(metadata: Dict) -> bool:
        expires_at = metadata.get("expires_at")
        if not expires_at:
            return False
        try:
            return datetime.now(timezone.utc) > datetime.fromisoformat(expires_at)
        except ValueError:
            return False

    def get_corpus(self, corpus_id: str) -> Dict:
        return self._read_json(self._metadata_path(corpus_id))

    def list_corpora(self) -> List[Dict]:
        corpora = []
        if self.use_gcs:
            blobs = self.client.list_blobs(self.bucket_name, prefix="corpora/")
            metadata_blob_names = [b.name for b in blobs if b.name.endswith("/metadata.json")]
            for name in metadata_blob_names:
                try:
                    corpora.append(self._read_json(name))
                except CorpusStoreError:
                    continue
        else:
            for child in self.local_root.iterdir():
                if not child.is_dir():
                    continue
                metadata_file = child / "metadata.json"
                if metadata_file.exists():
                    corpora.append(json.loads(metadata_file.read_text(encoding="utf-8")))

        corpora.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return corpora

    def _update_corpus(self, metadata: Dict) -> Dict:
        metadata["updated_at"] = self._now_iso()
        metadata["document_count"] = len(metadata.get("documents", []))
        self._write_json(self._metadata_path(metadata["corpus_id"]), metadata)
        return metadata

    def add_documents_bulk(self, corpus_id: str, documents: List[Dict]) -> Dict:
        """Upload multiple documents with a single metadata.json write.

        GCS rate-limits mutations to a single object to roughly 1/sec, so writing
        metadata.json once per document (as add_document does) triggers 429s when
        loading many files at once. Each document dict needs: document_id,
        original_filename, normalized_filename, file_type, size_bytes,
        normalized_text, and optionally parsing_status.
        """
        metadata = self.get_corpus(corpus_id)
        entries = []

        for doc in documents:
            storage_path = self._doc_blob_path(corpus_id, doc["normalized_filename"])
            self._upload_text(storage_path, doc["normalized_text"])
            entries.append({
                "document_id": doc["document_id"],
                "original_filename": doc["original_filename"],
                "normalized_filename": doc["normalized_filename"],
                "file_type": doc["file_type"],
                "uploaded_at": self._now_iso(),
                "size_bytes": int(doc["size_bytes"]),
                "parsing_status": doc.get("parsing_status", "parsed"),
                "storage_path": storage_path,
            })

        existing_documents = metadata.setdefault("documents", [])
        existing_documents.extend(entries)
        metadata["status"] = "needs_index"

        return self._update_corpus(metadata)

    def list_documents(self, corpus_id: str) -> List[Dict]:
        metadata = self.get_corpus(corpus_id)
        return metadata.get("documents", [])

    def delete_document(self, corpus_id: str, document_id: str) -> Dict:
        metadata = self.get_corpus(corpus_id)
        documents = metadata.get("documents", [])

        match = None
        keep = []
        for doc in documents:
            if doc.get("document_id") == document_id:
                match = doc
            else:
                keep.append(doc)

        if match is None:
            raise DocumentNotFoundError("Document not found")

        self._delete_path(match["storage_path"])
        metadata["documents"] = keep
        metadata["status"] = "needs_index"
        return self._update_corpus(metadata)

    def mark_ready(self, corpus_id: str) -> Dict:
        metadata = self.get_corpus(corpus_id)
        metadata["status"] = "ready"
        return self._update_corpus(metadata)

    def delete_corpus(self, corpus_id: str) -> None:
        metadata = self.get_corpus(corpus_id)
        for doc in metadata.get("documents", []):
            self._delete_path(doc["storage_path"])

        if self.use_gcs:
            prefix = f"{self._corpus_prefix(corpus_id)}/"
            for blob in self.client.list_blobs(self.bucket_name, prefix=prefix):
                blob.delete()
        else:
            local_dir = self.local_root / corpus_id
            if local_dir.exists():
                shutil.rmtree(local_dir)

    def delete_expired_corpora(self, kind: str = "demo") -> int:
        """Best-effort opportunistic cleanup; not a scheduled job."""
        deleted = 0
        for metadata in self.list_corpora():
            if metadata.get("kind") != kind:
                continue
            if not self.is_expired(metadata):
                continue
            try:
                self.delete_corpus(metadata["corpus_id"])
                deleted += 1
            except CorpusStoreError:
                continue
        return deleted

    def materialize_corpus(self, corpus_id: str, target_root: Optional[str] = None) -> Dict:
        metadata = self.get_corpus(corpus_id)
        documents = metadata.get("documents", [])

        if not documents:
            raise CorpusStoreError("Corpus has no documents")

        root = Path(target_root) if target_root else Path(tempfile.gettempdir()) / "govrag_corpora"
        local_dir = root / corpus_id

        if local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        filename_map = {}

        for doc in documents:
            normalized_filename = doc["normalized_filename"]
            storage_path = doc["storage_path"]
            text = self._download_text(storage_path)

            out_path = local_dir / normalized_filename
            out_path.write_text(text, encoding="utf-8")

            filename_map[normalized_filename] = doc.get("original_filename", normalized_filename)

        return {
            "local_dir": str(local_dir),
            "filename_map": filename_map,
            "document_count": len(documents),
            "updated_at": metadata.get("updated_at"),
            "status": metadata.get("status"),
        }
