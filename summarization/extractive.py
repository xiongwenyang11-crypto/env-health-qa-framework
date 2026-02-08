"""
summarization.extractive

Manuscript-aligned query-aware extractive summarization:

For each of the top-k retrieved documents:
- Split the document text into sentences
- Score each sentence by TF–IDF cosine similarity to the query
- Select the single highest-scoring sentence (n = 1 per document)

The final summary concatenates at most one sentence per document to preserve
traceability and avoid over-synthesis across heterogeneous sources.

This is a proof-of-concept implementation intended for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .sentence_split import sent_tokenize_simple


@dataclass
class SummarySentence:
    """
    A traceable unit for extractive summarization output.
    """
    doc_id: str
    title: str
    sentence: str
    sent_score: float
    doc_similarity: float


def _fit_sentence_vectorizer(sentences: List[str]) -> TfidfVectorizer:
    """
    Fit a lightweight TF–IDF vectorizer over candidate sentences.

    Kept intentionally simple and deterministic for reproducibility.
    Uses unigrams + english stopwords + L2 norm, consistent with manuscript.
    """
    v = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        norm="l2",
    )
    # If all sentences are empty/stopwords, sklearn may error; guard upstream.
    v.fit(sentences)
    return v


def _best_sentence_for_doc(
    query: str,
    text: str,
    *,
    min_sent_score: float = 0.0,
) -> Tuple[Optional[str], float]:
    """
    Select the single best sentence from a document based on TF–IDF cosine similarity.

    Returns:
        (best_sentence, best_score)
    """
    sents = [s.strip() for s in sent_tokenize_simple(text) if s and s.strip()]
    if not sents:
        return None, 0.0

    # Fit TF–IDF on sentences (document-internal scoring)
    try:
        v = _fit_sentence_vectorizer(sents + [query])
    except ValueError:
        # Happens if vocabulary is empty after stopword removal
        return None, 0.0

    sent_X = v.transform(sents)
    q_X = v.transform([query])

    sims = cosine_similarity(q_X, sent_X).ravel()
    if sims.size == 0:
        return None, 0.0

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score <= min_sent_score:
        return None, best_score

    return sents[best_idx], best_score


def summarize_retrieved(
    query: str,
    retrieved_df: pd.DataFrame,
    *,
    text_col: str = "text",
    doc_id_col: str = "doc_id",
    title_col: str = "title",
    similarity_col: str = "similarity",
    per_doc_n: int = 1,
    min_sent_score: float = 0.0,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate a manuscript-aligned extractive summary from retrieved documents.

    Expected retrieved_df columns (by default):
        - doc_id
        - title
        - similarity
        - text

    Behavior:
        - Select at most `per_doc_n` sentence(s) per document (default 1).
        - Preserves a one-to-one mapping between summarized statements and source docs.

    Returns:
        summary_text: concatenated selected sentences (doc order follows retrieval rank if present)
        selected_sentences_df: DataFrame with traceability columns:
            doc_id, title, similarity, sentence, sent_score
    """
    if retrieved_df is None or len(retrieved_df) == 0:
        return "No retrieved evidence available for summarization.", pd.DataFrame()

    for c in (doc_id_col, title_col, similarity_col, text_col):
        if c not in retrieved_df.columns:
            raise ValueError(f"Missing column '{c}' in retrieved_df.")

    if per_doc_n != 1:
        # Keep the API explicit: manuscript uses n=1 per document.
        # Allow changing only if intentionally experimenting.
        raise ValueError("per_doc_n must be 1 in the manuscript-aligned implementation.")

    rows: List[SummarySentence] = []

    # Preserve retrieval order if `rank` exists; otherwise keep row order
    if "rank" in retrieved_df.columns:
        doc_iter = retrieved_df.sort_values("rank").itertuples(index=False)
    else:
        doc_iter = retrieved_df.itertuples(index=False)

    for r in doc_iter:
        # Access by attribute names (safe with itertuples)
        doc_id = str(getattr(r, doc_id_col))
        title = str(getattr(r, title_col))
        doc_sim = float(getattr(r, similarity_col))
        text = getattr(r, text_col)
        text = "" if pd.isna(text) else str(text)

        best_sent, best_sc = _best_sentence_for_doc(query, text, min_sent_score=min_sent_score)
        if best_sent is None:
            continue

        rows.append(
            SummarySentence(
                doc_id=doc_id,
                title=title,
                sentence=best_sent,
                sent_score=float(best_sc),
                doc_similarity=doc_sim,
            )
        )

    if not rows:
        return "No query-relevant sentences were found in the retrieved evidence.", pd.DataFrame()

    out_df = pd.DataFrame([{
        "doc_id": s.doc_id,
        "title": s.title,
        "similarity": s.doc_similarity,
        "sentence": s.sentence,
        "sent_score": s.sent_score,
    } for s in rows])

    # For transparency, keep document order as retrieved; do not re-rank across documents
    summary_text = " ".join(out_df["sentence"].tolist())
    return summary_text, out_df
