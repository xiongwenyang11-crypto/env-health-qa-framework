"""
retrieval package API.

Re-exports the most commonly used retrieval utilities so that downstream code
can import from `retrieval` directly.

Manuscript alignment:
- Document-level TF–IDF indexing (title + abstract + key findings)
- Cosine-similarity retrieval returning ranked evidence items with scores/metadata
"""

from .tfidf_index import (
    TfidfIndex,
    build_document_text,
    build_tfidf_index,
    build_tfidf_index_from_dataframe,
)
from .cosine_search import retrieve_top_k

__all__ = [
    "TfidfIndex",
    "build_document_text",
    "build_tfidf_index",
    "build_tfidf_index_from_dataframe",
    "retrieve_top_k",
]