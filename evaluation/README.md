# evaluation module

This folder contains lightweight, reusable evaluation utilities used by `notebooks/evaluation_demo.ipynb`.

Modules:
- `metrics.py`: precision@k and per-scenario aggregation; overall summary statistics (mean ± SD)
- `agreement.py`: pairwise Cohen’s κ for inter-rater agreement
- `uncertainty.py`: majority-vote expert confidence and system–expert uncertainty alignment

The provided functions are designed to work with the CSV-based synthetic demo data in `data/demo_evaluation/` and can be reused with real evaluation tables by keeping the same schema.
