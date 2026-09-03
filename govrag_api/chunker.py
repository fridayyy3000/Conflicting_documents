"""Split long documents into overlapping chunks for retrieval.

Short documents (the common case for ConflictBench-style single-page bulletins)
are returned as a single chunk, so existing whole-document behavior is
unchanged when a document already fits within the target size.
"""

from __future__ import annotations

import re
from typing import List

# Word count is used as a cheap proxy for token count (~0.75 words/token is
# typical for English, so this comfortably targets the requested 800-1200
# token / 100-200 token overlap range without adding a tokenizer dependency).
TARGET_CHUNK_WORDS = 900
OVERLAP_WORDS = 150
MIN_CHUNK_WORDS = 50


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_text(
    text: str,
    target_words: int = TARGET_CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> List[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    if _word_count(text) <= target_words:
        return [text]

    # Paragraphs are the preferred unit; any paragraph bigger than the target
    # on its own is further split into sentences so nothing gets skipped.
    units: List[str] = []
    for para in _split_paragraphs(text):
        if _word_count(para) <= target_words:
            units.append(para)
        else:
            units.extend(_split_sentences(para))

    chunks: List[str] = []
    buffer_units: List[str] = []
    buffer_words = 0

    def flush() -> None:
        nonlocal buffer_units, buffer_words
        if buffer_units:
            chunks.append(" ".join(buffer_units).strip())
        buffer_units = []
        buffer_words = 0

    for unit in units:
        unit_words = _word_count(unit)

        if buffer_words + unit_words > target_words and buffer_units:
            flush()
            # Carry trailing overlap words forward from the chunk just finished.
            if chunks:
                tail_words = chunks[-1].split()[-overlap_words:]
                if tail_words:
                    buffer_units = [" ".join(tail_words)]
                    buffer_words = len(tail_words)

        buffer_units.append(unit)
        buffer_words += unit_words

    flush()

    # Merge a too-small trailing chunk into the previous one.
    if len(chunks) > 1 and _word_count(chunks[-1]) < MIN_CHUNK_WORDS:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks
