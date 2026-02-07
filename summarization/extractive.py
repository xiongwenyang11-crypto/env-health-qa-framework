"""
summarization.extractive

Query-aware extractive summarization:
- Split retrieved text(s) into sentences
- Score each sentence by overlap with query terms
- Select top-N sentences as the extractive summary

This is a proof-of-concept implementation intended for reproducibility.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import re
import numpy as np
import pandas as pd

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


def _normalize_terms(s: str) -> List[str]:
    s = s.lower()
    # keep letters/numbers; replace others with space
    s = re.sub(r"[^a-z0-9\s\-\.]", " ", s)
    terms = [t for t in s.split() if len(t) > 2]
    return terms


def _score_sentence(sentence: str, query_terms: set) -> float:
    sent_terms = set(_normalize_terms(sentence))
    if not sent_terms:
        return 0.0
    overlap = len(sent_terms.intersection(query_terms))
    # simple length normalization
    return float(overlap / np.sqrt(len(sent_terms)))


def summarize_retrieved(
    query: str,
    retrieved_df: pd.DataFrame,
    *,
    text_col: str = "text",
    doc_id_col: str = "doc_id",
    title_col: str = "title",
    similarity_col: str = "similarity",
    top_n_sentences: int = 3,
    min_sent_score: float = 0.0,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate an extractive summary from retrieved documents (DataFrame).

    Expected retrieved_df columns (by default):
        - doc_id
        - title
        - similarity
        - text

    Returns:
        summary_text: concatenated top sentences
        selected_sentences_df: DataFrame with traceability columns
            doc_id, title, similarity, sentence, sent_score
    """
    if retrieved_df is None or len(retrieved_df) == 0:
        return "No retrieved evidence available for summarization.", pd.DataFrame()

    for c in [doc_id_col, title_col, similarity_col, text_col]:
        if c not in retrieved_df.columns:
            raise ValueError(f"Missing column '{c}' in retrieved_df.")

    q_terms = set(_normalize_terms(query))
    candidates: List[SummarySentence] = []

    for _, r in retrieved_df.iterrows():
        doc_id = str(r[doc_id_col])
        title = str(r[title_col])
        sim = float(r[similarity_col])
        text = str(r[text_col]) if pd.notna(r[text_col]) else ""

        for sent in sent_tokenize_simple(text):
            sc = _score_sentence(sent, q_terms)
            if sc > min_sent_score:
                candidates.append(
                    SummarySentence(
                        doc_id=doc_id,
                        title=title,
                        sentence=sent,
                        sent_score=sc,
                        doc_similarity=sim,
                    )
                )

    if not candidates:
        return "No overlapping evidence sentences found for this query.", pd.DataFrame()

    # Rank primarily by sentence score, secondarily by doc similarity
    cdf = pd.DataFrame([{
        "doc_id": c.doc_id,
        "title": c.title,
        "similarity": c.doc_similarity,
        "sentence": c.sentence,
        "sent_score": c.sent_score,
    } for c in candidates])

    cdf = cdf.sort_values(["sent_score", "similarity"], ascending=False)

    selected = cdf.head(top_n_sentences).reset_index(drop=True)
    summary_text = " ".join(selected["sentence"].tolist())
    return summary_text, selected
