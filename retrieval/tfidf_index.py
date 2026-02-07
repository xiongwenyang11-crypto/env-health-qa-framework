"""
TF–IDF indexing utilities for an interpretable, evidence-based QA pipeline.

This module is intentionally minimal and reproducible:
- Builds a TF–IDF index over a list of document texts
- Stores doc_ids and titles aligned with the TF–IDF matrix rows
- Designed to be used by retrieval/cosine_search.py

No external services or black-box models are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class TfidfIndex:
    """
    Container holding a TF–IDF representation of a document collection.

    Attributes
    ----------
    vectorizer:
        Fitted sklearn TfidfVectorizer.
    X:
        TF–IDF document-term matrix, shape (n_docs, n_features).
    doc_ids:
        Document identifiers aligned with rows of X.
    titles:
        Document titles aligned with rows of X.
    """

    vectorizer: TfidfVectorizer
    X: csr_matrix
    doc_ids: List[str]
    titles: List[str]


def _validate_inputs(
    texts: List[str],
    doc_ids: Optional[List[str]],
    titles: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    if texts is None or len(texts) == 0:
        raise ValueError("`texts` must be a non-empty list of document strings.")

    n = len(texts)

    if doc_ids is None:
        doc_ids = [f"D{i+1}" for i in range(n)]
    if titles is None:
        titles = [f"Document {i+1}" for i in range(n)]

    if len(doc_ids) != n:
        raise ValueError(f"`doc_ids` length ({len(doc_ids)}) must match texts length ({n}).")
    if len(titles) != n:
        raise ValueError(f"`titles` length ({len(titles)}) must match texts length ({n}).")

    # Ensure all entries are strings (helps avoid surprise serialization issues)
    doc_ids = [str(x) for x in doc_ids]
    titles = [str(x) for x in titles]

    return doc_ids, titles


def build_tfidf_index(
    *,
    texts: List[str],
    doc_ids: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    lowercase: bool = True,
    stop_words: Optional[Union[str, List[str]]] = "english",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = 5000,
    min_df: Union[int, float] = 1,
    max_df: Union[int, float] = 1.0,
) -> TfidfIndex:
    """
    Build a TF–IDF index for a set of documents.

    Parameters
    ----------
    texts:
        List of document texts, one per document.
    doc_ids:
        Optional list of document IDs aligned with texts.
        If None, IDs are generated as D1, D2, ...
    titles:
        Optional list of human-readable titles aligned with texts.
        If None, titles are generated as "Document 1", "Document 2", ...
    lowercase:
        Whether to lowercase during vectorization.
    stop_words:
        Stop words to remove. Use "english" for a reasonable default.
    ngram_range:
        N-gram range for TF–IDF features.
    max_features:
        Max number of features to keep (None means no limit).
    min_df, max_df:
        Document-frequency thresholds passed through to sklearn.

    Returns
    -------
    TfidfIndex
        A lightweight container with the fitted vectorizer, TF–IDF matrix, and metadata.

    Notes
    -----
    - This function intentionally does NOT implement any fancy preprocessing beyond what
      sklearn provides, to keep the pipeline transparent and reproducible.
    - For paper consistency, keep your vectorizer settings aligned with the manuscript.
    """
    doc_ids, titles = _validate_inputs(texts=texts, doc_ids=doc_ids, titles=titles)

    vectorizer = TfidfVectorizer(
        lowercase=lowercase,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    X = vectorizer.fit_transform(texts)

    # Sanity checks (defensive, helpful for reviewers/users)
    if X.shape[0] != len(texts):
        raise RuntimeError("TF–IDF matrix row count does not match input texts.")
    if X.shape[1] == 0:
        raise RuntimeError(
            "TF–IDF produced 0 features. Check stop_words/min_df/max_df settings."
        )

    return TfidfIndex(vectorizer=vectorizer, X=X, doc_ids=doc_ids, titles=titles)
