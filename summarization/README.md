# summarization module

This folder implements a lightweight, query-aware extractive summarization method for demonstration and reproducibility.

Approach:
- sentence splitting (simple punctuation-based)
- sentence scoring based on query-term overlap (length-normalized)
- select top-N sentences across retrieved documents

Main API:
- `summarize_retrieved(query, retrieved_df, ...)` in `extractive.py`
