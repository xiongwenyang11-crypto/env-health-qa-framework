"""
retrieval.cosine_search

Cosine-similarity search over a TF–IDF index.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .tfidf_index import TfidfIndex


def retrieve_top_k(
    query: str,
    index: TfidfIndex,
    k: int = 5,
    *,
    include_text: bool = True,
    snippet_len: int = 360,
) -> pd.DataFrame:
    """
    Retrieve top-k documents for a query using cosine similarity.

    Returns a DataFrame with:
        - rank
        - doc_id
        - title
        - similarity
        - snippet
        - text (optional, controlled by include_text)

    Notes:
        - This is intended for demo / proof-of-concept reproducibility.
        - For real systems, consider chunk-level indexing.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if k <= 0:
        raise ValueError("k must be positive.")

    q_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(q_vec, index.doc_matrix).ravel()

    k_eff = min(k, len(index.doc_ids))
    top_idx = np.argsort(-sims)[:k_eff]

    rows = []
    for rank, i in enumerate(top_idx, start=1):
        title = index.titles[i] if index.titles is not None else index.doc_ids[i]
        text = index.texts[i] if index.texts is not None else ""

        snippet_src = " ".join(text.split())
        snippet = snippet_src[:snippet_len] + ("..." if len(snippet_src) > snippet_len else "")

        row = {
            "rank": rank,
            "doc_id": index.doc_ids[i],
            "title": title,
            "similarity": float(sims[i]),
            "snippet": snippet,
        }
        if include_text:
            row["text"] = text

        rows.append(row)

    return pd.DataFrame(rows)
