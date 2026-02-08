# summarization module

This folder implements a transparent, query-aware extractive summarization
component used as a methodological surrogate for LLM-style reasoning in the
proof-of-concept pipeline.

The summarization process is intentionally non-generative and fully
traceable, preserving an explicit one-to-one correspondence between
summarized statements and their source documents.

---

## Manuscript-Aligned Approach

The extractive summarization follows the methodology described in the manuscript:

- Retrieved documents are processed **individually**.
- Each document text is split into sentence-level segments using a lightweight,
  deterministic sentence splitter.
- Sentences within each document are scored using **TF–IDF cosine similarity**
  with respect to the user query.
- For each retrieved document, **only the single highest-scoring sentence**
  (n = 1 per document) is selected.
- The final summary concatenates at most one sentence per document, preserving
  traceability and avoiding over-synthesis across heterogeneous sources.

This design prioritizes interpretability, reproducibility, and evidence
traceability over semantic abstraction or fluency.

---

## Files

- `sentence_split.py`  
  Lightweight sentence segmentation utilities that handle punctuation,
  newlines, and section headers commonly found in scientific abstracts
  and structured study descriptions.

- `extractive.py`  
  Query-aware extractive summarization logic that selects one representative,
  query-relevant sentence per retrieved document using TF–IDF cosine similarity.

---

## Main API

- `summarize_retrieved(query, retrieved_df, ...)`  
  Generates a citation-grounded extractive summary together with a
  traceability table linking each selected sentence to its source document.

---

## Scope and Intended Use

This module is designed for methodological demonstration and expert inspection
in a proof-of-concept setting. It is not intended to perform semantic reasoning,
causal inference, or abstractive text generation, and it does not replace
expert interpretation of the underlying evidence.
