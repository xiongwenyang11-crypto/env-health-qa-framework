"""
TF–IDF indexing utilities for an interpretable, evidence-based QA pipeline.

Manuscript alignment:
- Indexing is performed at the document level.
- Each document is represented by concatenating: title + abstract + key findings.
- Retrieval uses TF–IDF (unigrams, stop-word removal, L2 normalization) + cosine similarity.

This module is intentionally minimal and reproducible:
- Builds a TF–IDF index over a list of document texts
- Stores doc_ids and optional titles/texts/metadata aligned with TF–IDF matrix rows
- Designed to be used by retrieval/cosine_search.py

No external services or black-box models are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfIndex:
    """
    Container holding a TF–IDF representation of a document collection.

    Attributes
    ----------
    vectorizer:
        Fitted sklearn TfidfVectorizer.
    doc_matrix:
        TF–IDF document-term matrix, shape (n_docs, n_features).
    doc_ids:
        Document identifiers aligned with rows of doc_matrix.
    titles:
        Optional document titles aligned with rows of doc_matrix.
    texts:
        Optional raw document texts aligned with rows of doc_matrix (used for snippets / inspection).
    metadata_df:
        Optional metadata table keyed by doc_id for downstream display (evidence cards).
    """

    vectorizer: TfidfVectorizer
    doc_matrix: csr_matrix
    doc_ids: List[str]
    titles: Optional[List[str]] = None
    texts: Optional[List[str]] = None
    metadata_df: Optional[pd.DataFrame] = None


def _validate_equal_length(name: str, xs: Sequence, n: int) -> None:
    if xs is None:
        return
    if len(xs) != n:
        raise ValueError(f"`{name}` length ({len(xs)}) must match number of documents ({n}).")


def build_document_text(
    *,
    title: str = "",
    abstract: str = "",
    key_findings: str = "",
) -> str:
    """
    Build a single document text representation consistent with the manuscript:
    concatenation of title + abstract + key findings.

    This function is intentionally simple and transparent.
    """
    parts = []
    for p in (title, abstract, key_findings):
        p = "" if p is None else str(p).strip()
        if p:
            parts.append(p)
    return "\n\n".join(parts)


def build_tfidf_index(
    *,
    texts: List[str],
    doc_ids: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    lowercase: bool = True,
    stop_words: Optional[Union[str, List[str]]] = "english",
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: Optional[int] = None,
    min_df: Union[int, float] = 1,
    max_df: Union[int, float] = 1.0,
    norm: str = "l2",
) -> TfidfIndex:
    """
    Build a TF–IDF index for a set of documents.

    Parameters
    ----------
    texts:
        List of document texts, one per document (already concatenated as needed).
    doc_ids:
        Optional list of document IDs aligned with texts.
        If None, IDs are generated as D1, D2, ...
    titles:
        Optional list of titles aligned with texts.
    metadata_df:
        Optional metadata DataFrame with a 'doc_id' column to merge later.
    lowercase, stop_words, ngram_range, max_features, min_df, max_df, norm:
        Passed through to sklearn's TfidfVectorizer.

    Returns
    -------
    TfidfIndex
        Lightweight container with fitted vectorizer, TF–IDF matrix, and aligned metadata.

    Notes
    -----
    - This function intentionally avoids complex preprocessing to remain transparent and reproducible.
    - For manuscript consistency, keep defaults aligned with the paper:
        unigram TF–IDF, stop-word removal, L2 normalization.
    """
    if texts is None or len(texts) == 0:
        raise ValueError("`texts` must be a non-empty list of document strings.")

    n = len(texts)

    if doc_ids is None:
        doc_ids = [f"D{i+1}" for i in range(n)]
    doc_ids = [str(x) for x in doc_ids]
    _validate_equal_length("doc_ids", doc_ids, n)

    if titles is not None:
        titles = [str(x) for x in titles]
        _validate_equal_length("titles", titles, n)

    if metadata_df is not None:
        if not isinstance(metadata_df, pd.DataFrame):
            raise ValueError("metadata_df must be a pandas DataFrame.")
        if "doc_id" not in metadata_df.columns:
            raise ValueError("metadata_df must contain a 'doc_id' column.")
        # keep only rows that correspond to doc_ids (optional, but prevents accidental mismatches)
        metadata_df = metadata_df.copy()
        metadata_df["doc_id"] = metadata_df["doc_id"].astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=lowercase,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        norm=norm,
    )
    doc_matrix = vectorizer.fit_transform(texts)

    if doc_matrix.shape[0] != n:
        raise RuntimeError("TF–IDF matrix row count does not match input texts.")
    if doc_matrix.shape[1] == 0:
        raise RuntimeError("TF–IDF produced 0 features. Check stop_words/min_df/max_df settings.")

    return TfidfIndex(
        vectorizer=vectorizer,
        doc_matrix=doc_matrix,
        doc_ids=doc_ids,
        titles=titles,
        texts=texts,
        metadata_df=metadata_df,
    )


def build_tfidf_index_from_dataframe(
    docs: pd.DataFrame,
    *,
    doc_id_col: str = "doc_id",
    title_col: str = "title",
    abstract_col: str = "abstract",
    key_findings_col: str = "key_findings",
    keep_metadata_cols: Optional[List[str]] = None,
    **vectorizer_kwargs,
) -> TfidfIndex:
    """
    Convenience constructor: build TF–IDF index directly from a document table.

    Expected columns (manuscript-aligned):
    - doc_id
    - title
    - abstract
    - key_findings

    Any additional columns can be stored as metadata for evidence cards.

    Returns:
        TfidfIndex
    """
    for c in (doc_id_col, title_col, abstract_col, key_findings_col):
        if c not in docs.columns:
            raise KeyError(f"Missing column '{c}' in docs dataframe.")

    df = docs.copy()
    df[doc_id_col] = df[doc_id_col].astype(str)

    texts = [
        build_document_text(
            title=row.get(title_col, ""),
            abstract=row.get(abstract_col, ""),
            key_findings=row.get(key_findings_col, ""),
        )
        for _, row in df.iterrows()
    ]
    doc_ids = df[doc_id_col].tolist()
    titles = df[title_col].astype(str).tolist()

    # Metadata: keep selected cols or everything except text fields
    if keep_metadata_cols is None:
        drop_cols = {title_col, abstract_col, key_findings_col}
        meta_cols = [c for c in df.columns if c not in drop_cols]
    else:
        meta_cols = [doc_id_col] + [c for c in keep_metadata_cols if c != doc_id_col]

    metadata_df = df[meta_cols].copy()

    return build_tfidf_index(
        texts=texts,
        doc_ids=doc_ids,
        titles=titles,
        metadata_df=metadata_df,
        **vectorizer_kwargs,
    )