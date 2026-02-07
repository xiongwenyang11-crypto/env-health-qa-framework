"""
summarization.sentence_split

Lightweight sentence splitting utilities without external NLP dependencies.
This is sufficient for demonstration purposes and reproducibility.
"""

from __future__ import annotations
import re
from typing import List


_SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def sent_tokenize_simple(text: str) -> List[str]:
    """
    A simple sentence splitter based on punctuation boundaries.

    Args:
        text: input text

    Returns:
        list of sentence strings (non-empty)
    """
    if not text:
        return []
    parts = _SENT_SPLIT_REGEX.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]
