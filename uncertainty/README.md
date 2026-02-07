# uncertainty module

This folder provides a lightweight uncertainty calibration mechanism for the demo pipeline.

Workflow:
1) take raw retrieval similarity scores
2) optional min-max normalization
3) sigmoid transformation
4) threshold-based mapping to Low / Medium / High

Main API:
- `annotate_retrieval_df(...)`: adds calibrated score + label to retrieval output
- `overall_confidence(...)`: derives an overall confidence label for the answer
