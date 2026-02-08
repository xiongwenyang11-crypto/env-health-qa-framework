# evaluation module

This folder contains lightweight, reusable evaluation utilities used by
`notebooks/evaluation_demo.ipynb` to reproduce the proof-of-concept evaluation
described in the accompanying manuscript.

The evaluation design emphasizes interpretability, transparency, and expert-aligned
assessment rather than benchmark-style performance optimization.

---

## Modules

- `precision_at_k.py`  
  Utilities for computing citation precision@k, defined as the proportion of
  expert-judged relevant studies among the top-k retrieved items.  
  Supports both document-level (long-format) relevance labels and legacy list-based
  representations for demonstration purposes.

- `agreement.py`  
  Pairwise inter-rater agreement utilities based on Cohen’s κ, used to characterize
  consistency among expert reviewers for ordinal or categorical ratings
  (e.g., factual consistency, interpretability).

- `uncertainty.py`  
  Utilities for expert confidence aggregation (majority vote) and uncertainty
  alignment between system-assigned confidence levels and expert consensus confidence.
  Alignment is quantified using weighted Cohen’s κ with linear weights, treating
  Low/Medium/High as an ordinal scale, in accordance with the manuscript.

- `interpretability_stats.py`  
  Aggregation and reporting utilities for building per-scenario evaluation tables
  and summary statistics (mean ± SD), as well as exporting reproducible CSV outputs
  used in the Results section.

---

## Data Compatibility

All functions are designed to operate on CSV-based **synthetic demonstration data**
provided in `data/demo_evaluation/`. These data illustrate the evaluation workflow
only and do not represent real expert judgments or real study outputs.

The same functions can be applied to real evaluation tables by preserving the
expected column schemas (e.g., scenario identifiers, expert identifiers,
document-level relevance labels, and confidence categories).

---

## Intended Use

This evaluation module is intended for research and methodological demonstration
purposes only. It supports transparent inspection of system behavior and expert
alignment in a controlled proof-of-concept setting and is not designed to establish
generalizable performance benchmarks or decision-support validity.