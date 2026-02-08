"""
retrieval.cosine_search

Cosine-similarity search over a TF–IDF index.

Manuscript alignment:
- Retrieval maps a user query to a ranked list of evidence items with similarity scores and metadata.
- The demo/proof-of-concept uses deterministic TF–IDF + cosine similarity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .tfidf_index import TfidfIndex


def retrieve_top_k(
    query: str,
    index: TfidfIndex,
    k: int = 3,
    *,
    include_text: bool = False,
    snippet_len: int = 360,
    include_snippet: bool = True,
) -> pd.DataFrame:
    """
    Retrieve top-k documents for a query using cosine similarity.

    Parameters
    ----------
    query:
        Natural-language user query.
    index:
        TfidfIndex containing vectorizer, doc_matrix, doc_ids, and optional metadata/text.
    k:
        Number of documents to retrieve (default 3, consistent with the manuscript).
    include_text:
        If True, include full text in the output (off by default to reduce payload size).
    snippet_len:
        Max snippet length if snippets are included.
    include_snippet:
        If True, include a whitespace-normalized snippet derived from text (if available).

    Returns
    -------
    pd.DataFrame
        Columns (minimum):
          - rank            (1..k_returned)
          - doc_id
          - similarity
          - k_requested
          - k_returned
        Optional (if available):
          - title
          - snippet
          - text
        Optional metadata (if index exposes it):
          - any metadata columns merged by doc_id

    Notes
    -----
    - Deterministic: fixed vectorizer + fixed corpus -> reproducible rankings and scores.
    - Sorting is stable: ties are broken by doc_id for reproducibility.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if k <= 0:
        raise ValueError("k must be positive.")

    if index.doc_matrix is None or index.vectorizer is None:
        raise ValueError("index must contain a fitted vectorizer and doc_matrix.")

    q_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(q_vec, index.doc_matrix).ravel()

    n_docs = len(index.doc_ids)
    k_eff = int(min(k, n_docs))

    # Stable sort: primary = -similarity, secondary = doc_id (string)
    doc_ids = np.asarray(index.doc_ids, dtype=object)
    tie_break = np.argsort(doc_ids.astype(str))  # consistent ordering for doc_id ties
    # Apply tie-break ordering first, then stable sort by similarity
    sims_tb = sims[tie_break]
    ids_tb = doc_ids[tie_break]
    order_tb = np.argsort(-sims_tb, kind="mergesort")[:k_eff]
    top_idx = tie_break[order_tb]

    rows = []
    for rank, i in enumerate(top_idx, start=1):
        doc_id = index.doc_ids[i]
        title = index.titles[i] if getattr(index, "titles", None) is not None else doc_id

        text = ""
        if getattr(index, "texts", None) is not None and index.texts is not None:
            text = index.texts[i] or ""

        row = {
            "rank": rank,
            "doc_id": doc_id,
            "title": title,
            "similarity": float(sims[i]),
            "k_requested": int(k),
            "k_returned": int(k_eff),
        }

        if include_snippet:
            snippet_src = " ".join(text.split()) if text else ""
            if snippet_src:
                row["snippet"] = snippet_src[:snippet_len] + ("..." if len(snippet_src) > snippet_len else "")
            else:
                row["snippet"] = ""

        if include_text:
            row["text"] = text

        rows.append(row)

    out = pd.DataFrame(rows)

    # Optional: merge metadata if index exposes a metadata table keyed by doc_id
    meta = getattr(index, "metadata_df", None)
    if isinstance(meta, pd.DataFrame) and "doc_id" in meta.columns:
        out = out.merge(meta, on="doc_id", how="left")

    return out