"""
summarization module API

This module re-exports the core components of the manuscript-aligned
query-aware extractive summarization pipeline, including sentence splitting
and document-level summary generation.

Modules:
- sentence_split.py: Simple, reproducible sentence splitting with manual handling of section headers.
- extractive.py: Query-aware extractive summarization, selecting the top-scoring sentence per document.
"""

from .sentence_split import sent_tokenize_simple
from .extractive import summarize_retrieved

__all__ = ["sent_tokenize_simple", "summarize_retrieved"]
