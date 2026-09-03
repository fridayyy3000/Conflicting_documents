"""Document parsing and normalization utilities for corpus ingestion."""

from __future__ import annotations

import io
import os
import re
from typing import Tuple

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed safely."""


def sanitize_filename(filename: str) -> str:
    """Sanitize an incoming filename to a safe basename."""
    base = os.path.basename((filename or "").strip())
    if not base:
        raise DocumentParseError("Filename is required")

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    safe = safe.strip("._")
    if not safe:
        raise DocumentParseError("Filename is invalid after sanitization")

    return safe


def validate_extension(filename: str) -> str:
    """Validate supported extension and return normalized extension."""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentParseError(f"Unsupported file type '{ext}'. Supported: {supported}")
    return ext


def parse_document_bytes(content: bytes, extension: str) -> str:
    """Extract normalized text from supported file bytes."""
    if not content:
        raise DocumentParseError("File is empty")

    if extension in {".md", ".txt"}:
        return content.decode("utf-8", errors="replace").strip()

    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()

    if extension == ".docx":
        doc = DocxDocument(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paragraphs).strip()

    raise DocumentParseError(f"Parser not implemented for extension '{extension}'")


def make_normalized_filename(safe_original_filename: str, document_id: str) -> Tuple[str, str]:
    """Build a unique normalized markdown filename and stem."""
    stem, _ = os.path.splitext(safe_original_filename)
    normalized_stem = f"{stem}__{document_id}"
    normalized_filename = f"{normalized_stem}.md"
    return normalized_stem, normalized_filename
