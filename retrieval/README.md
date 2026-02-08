# retrieval module

This folder implements a transparent and reproducible document-level retrieval
module based on TF–IDF vectorization and cosine similarity.

The retrieval component is designed to support evidence-centered question
answering by mapping a natural-language query to a ranked set of literature
items with explicit similarity scores and inspectable metadata.

---

## Design Overview

- Retrieval is performed at the **document level**.
- Each document is indexed using a text representation formed by concatenating
  **title, abstract, and manually curated key findings**, consistent with the
  manuscript.
- Relevance ranking is computed using TF–IDF (unigrams, stop-word removal,
  L2 normalization) followed by cosine similarity.
- Similarity scores are retained and exposed to downstream components for
  transparency and qualitative uncertainty calibration.

The implementation is deterministic and intentionally lightweight, prioritizing
interpretability and reproducibility over retrieval performance optimization.

---

## Files

- `tfidf_index.py`  
  Utilities for building a TF–IDF index over a collection of documents, including
  helper functions for constructing document texts and attaching structured
  metadata for evidence inspection.

- `cosine_search.py`  
  Query-time cosine similarity search that returns a ranked list of top-*k*
  documents with similarity scores, snippets, and optional metadata, suitable
  for interactive evidence inspection and evaluation.

---

## Scope and Intended Use

This module represents a proof-of-concept implementation aligned with the
retrieval component described in the manuscript. It is intended for methodological
demonstration and reproducible experimentation and is not designed to serve as a
production-grade or performance-optimized retrieval system.
