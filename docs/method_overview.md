## Method Overview

This document provides a high-level overview of the design rationale and methodological structure of the interpretable environmental health question-answering framework implemented in this repository.

The framework is designed to support transparent, evidence-centered synthesis of environmental health literature, with explicit uncertainty representation and full traceability to source evidence, consistent with the accompanying manuscript.

## Design Principles

The framework is guided by the following core principles:

Evidence-centered reasoning
All system outputs are grounded in explicitly retrieved literature evidence rather than unconstrained text generation.

Interpretability over black-box prediction
Each stage of the pipeline is explicitly defined and inspectable, avoiding opaque end-to-end models.

Traceability and transparency
Users can trace system outputs to specific source studies and text passages.

Explicit uncertainty representation
Qualitative confidence levels are treated as first-class outputs rather than implicit by-products.

Reproducibility
The system relies on deterministic methods and standard Python libraries, with demonstration datasets provided to support reproducibility.

## System Architecture

The framework follows a modular, query-driven pipeline architecture consisting of four core components:

Knowledge Representation and Curation

Evidence Retrieval

Interpretable Reasoning with Uncertainty Calibration

Evaluation and Interactive Inspection

Each component is implemented as an independent module to support clarity, extensibility, and reproducibility.

## Evidence Retrieval

The retrieval module identifies literature relevant to a user’s natural-language query.

Documents are indexed using TF–IDF vector representations.

Cosine similarity is used to rank documents by relevance.

The top-k documents (k = 3 in the reported study) are returned together with similarity scores and structured metadata.

This design prioritizes interpretability and deterministic behavior over complex neural retrieval models that may obscure how relevance scores are produced.

For demonstration purposes, retrieval is performed at the document level. The architecture is extensible to sentence- or paragraph-level retrieval in future work.

## Interpretable Extractive Reasoning

Retrieved documents are summarized using a query-aware extractive aggregation strategy designed to preserve traceability.

Each retrieved document is segmented into sentences using lightweight rule-based procedures.

Sentence-level TF–IDF cosine similarity scores are computed between the query and all sentences within each document.

Exactly one sentence (n = 1) is selected per retrieved document, corresponding to the highest-scoring sentence for that document.

This one-to-one correspondence between documents and extracted sentences ensures that each summarized statement can be directly traced to a specific source study and limits over-synthesis across heterogeneous evidence.

## Uncertainty Calibration

Uncertainty is represented using qualitative confidence levels derived from retrieval-based evidence support signals.

The calibration procedure follows three deterministic steps:

Min–max normalization of document-level similarity scores from the top-k retrieved studies.

Aggregation using the arithmetic mean to produce a single query-level evidence support indicator.

Threshold-based mapping of the aggregated score into three ordinal categories:

Low

Medium

High

Thresholds are fixed prior to evaluation and applied uniformly across all queries.

This approach provides an interpretable proxy for relative evidence support, rather than probabilistic estimates of correctness, effect size, or causal certainty.

## Evaluation Strategy

The evaluation framework assesses both factual consistency and interpretability of system outputs.

Key evaluation dimensions include:

Citation precision@k
Proportion of expert-judged relevant studies among the top-k retrieved documents.

Factual consistency
Expert judgment using a three-level ordinal scale (0–2) reflecting incorrect, partially consistent, or fully consistent summaries.

Interpretability
Expert-rated clarity and transparency using a 5-point Likert scale.

Uncertainty alignment
Agreement between system-assigned qualitative confidence and expert-perceived confidence, assessed using weighted Cohen’s κ.

Inter-rater agreement
Reported descriptively to contextualize expert judgment variability.

Synthetic demonstration data included in this repository are used solely to illustrate evaluation logic and metric computation.

## Demonstration Data vs. Real Literature

This repository includes synthetic demonstration datasets intended to illustrate the pipeline and evaluation procedures.

Demonstration texts do not represent the full literature used in the study.

Text content is simplified or synthetic to avoid copyright restrictions.

Evaluation CSV files contain synthetic expert ratings for reproducibility only.

The data schemas and code structure support direct substitution with real literature and expert evaluations.

## Scope and Extensions

This implementation represents a proof-of-concept framework rather than a production system.

Planned extensions include:

Integration with large-scale bibliographic databases

Embedding-based retrieval backends

Automated entity recognition

Citation-constrained generative reasoning modules

These extensions are intentionally beyond the scope of the current implementation and manuscript.

## Intended Use

This framework and accompanying code are intended for research and educational purposes only.

They are not designed to replace expert judgment or to support clinical or public health decision-making.