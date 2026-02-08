# uncertainty module

This folder implements a transparent and interpretable uncertainty representation
mechanism for the evidence-centered question answering pipeline.

Uncertainty is treated as a qualitative indicator of relative evidence support,
rather than as a probabilistic estimate of correctness or causal certainty.

---

## Manuscript-Aligned Workflow

The uncertainty computation follows the methodology described in the manuscript:

1) Collect similarity scores from the top-k retrieved documents (k = 3 in the demo)
2) Apply min–max normalization to rescale scores to [0, 1]
3) Aggregate normalized scores using the arithmetic mean to obtain a single
   evidence-support indicator
4) Map the support indicator to qualitative confidence levels
   (**Low / Medium / High**) using fixed thresholds defined prior to evaluation

This process is deterministic and fully reproducible given fixed inputs and
thresholds.

---

## Main APIs

- `compute_overall_confidence_from_retrieval(...)`  
  Primary query-level function that computes both the evidence-support indicator
  and the corresponding qualitative confidence label from retrieval results.

- `compute_support_indicator(...)`  
  Computes the aggregated evidence-support score (mean of normalized similarities).

- `annotate_retrieval_df(...)`  
  Optional helper that annotates retrieval outputs with per-document normalized
  similarity scores for inspection and visualization purposes.  
  (This does not define the final query-level confidence by itself.)

---

## Design Notes

- The uncertainty labels (**Low / Medium / High**) are intended as interpretable
  communicative cues, not calibrated probabilities.
- No sigmoid transformation or learned calibration is used in the manuscript-
  aligned workflow.
- Thresholds are fixed prior to evaluation to support transparency and
  reproducibility.

---

## Scope and Intended Use

This module is designed for methodological demonstration and research use.
It supports cautious interpretation of heterogeneous environmental health
evidence and is not intended to replace expert judgment or to function as a
decision-support system.