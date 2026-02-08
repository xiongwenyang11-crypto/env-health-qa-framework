"""
summarization.sentence_split

Lightweight sentence splitting utilities without external NLP dependencies.

Manuscript alignment:
- Deterministic and reproducible preprocessing.
- Suitable for proof-of-concept corpora where texts may contain section headers
  (e.g., "Population:", "Exposure:", "Key findings:") and newline-separated bullets.

This splitter treats both punctuation boundaries and newlines as candidate splits,
and merges standalone section headers with the following content line to avoid
selecting meaningless header-only "sentences".
"""

from __future__ import annotations

import re
from typing import List


# Split on punctuation followed by whitespace OR on newlines.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# A simple pattern for section headers like "Population:" or "Key findings:"
_SECTION_HEADER = re.compile(r"^[A-Za-z][A-Za-z\s\-/]*:\s*$")


def sent_tokenize_simple(text: str) -> List[str]:
    """
    Deterministic sentence/segment splitter.

    Rules:
    1) Split on punctuation boundaries (. ! ?) and on newlines.
    2) Strip whitespace and drop empty segments.
    3) If a segment is a standalone "Header:" line, merge it with the next segment.

    Args:
        text: input text

    Returns:
        list of sentence/segment strings (non-empty)
    """
    if not text or not str(text).strip():
        return []

    raw_parts = [p.strip() for p in _SENT_BOUNDARY.split(str(text).strip()) if p and p.strip()]
    if not raw_parts:
        return []

    merged: List[str] = []
    i = 0
    while i < len(raw_parts):
        cur = raw_parts[i]

        # Merge standalone headers with the next segment (if any)
        if _SECTION_HEADER.match(cur) and (i + 1) < len(raw_parts):
            nxt = raw_parts[i + 1]
            merged.append(f"{cur} {nxt}".strip())
            i += 2
            continue

        merged.append(cur)
        i += 1

    return merged
