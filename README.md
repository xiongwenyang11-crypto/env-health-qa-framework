Interpretable Environmental Health Question-Answering Framework
===============================================================

This repository provides a proof-of-concept implementation of an interpretable,
evidence-centered question-answering framework for environmental health research.

The framework is designed to support transparent synthesis of pollutant–health
evidence from scientific literature, with explicit uncertainty representation
and full traceability to source studies.

The code accompanies the academic manuscript:

“Development and Evaluation of a Transparent AI Framework for Evidence-Centered
Question Answering in Environmental Health Research.”

---

## Overview

Environmental health evidence is often dispersed across heterogeneous studies,
disciplines, and exposure contexts, making synthesis and inspection challenging.
This framework explores how AI-inspired retrieval and reasoning techniques can be
combined with structured, transparent workflows to support:

- Natural-language question answering on pollutant–health relationships  
- Evidence-centered, citation-grounded responses  
- Explicit and interpretable representation of uncertainty  
- Transparent linkage between system outputs and source literature  

Rather than relying on end-to-end or black-box large language model outputs, the
framework prioritizes **interpretability, reproducibility, and evidence
traceability** as first-class design principles.

The repository implements a lightweight, non-generative prototype intended to
demonstrate **methodological feasibility** rather than a production-level
decision-support system.

---

## Framework Components

The repository structure reflects the main methodological components described
in the manuscript:

### Evidence Retrieval

Relevant literature is retrieved using TF–IDF vectorization and cosine
similarity. Queries are mapped to a ranked list of candidate studies together
with transparent similarity scores and study metadata.

### Extractive Summarization

Retrieved documents are summarized using a **query-aware extractive strategy**.
Each document is split into sentences, and sentence-level relevance to the query
is scored using TF–IDF cosine similarity.  
For each retrieved document, **a single highest-scoring sentence (n = 1 per
document)** is selected, ensuring a one-to-one correspondence between summarized
statements and source studies.

This design prioritizes traceability and avoids over-synthesis across
heterogeneous evidence sources.

### Uncertainty Representation

Uncertainty is represented using a qualitative confidence indicator
(**Low / Medium / High**) derived from retrieval-based evidence support signals.

Specifically:
- Similarity scores from the top-k retrieved documents are min–max normalized
- Normalized scores are aggregated using their arithmetic mean
- Fixed thresholds, defined prior to evaluation, are applied to assign
  qualitative confidence levels

These labels are intended as interpretable communicative cues rather than
probabilistic estimates of correctness or causal certainty.

### Evaluation

The repository includes scripts and notebooks to reproduce the evaluation
workflow reported in the manuscript, including:

- Citation precision@k  
- Expert-assessed factual consistency and interpretability  
- Inter-rater agreement (pairwise Cohen’s κ)  
- Alignment between system-assigned uncertainty labels and expert consensus
  confidence (weighted Cohen’s κ with linear weights)

Evaluation focuses on **internal consistency, interpretability, and transparency**
rather than benchmark performance or generalizable accuracy.

---

## Repository Structure

env-health-qa-framework/
│
├── data/
│ ├── demo_corpus/ # Synthetic demonstration studies
│ └── demo_evaluation/ # Synthetic evaluation tables
│
├── retrieval/ # TF–IDF indexing and cosine-similarity search
│
├── summarization/ # Query-aware extractive summarization
│
├── uncertainty/ # Qualitative uncertainty representation utilities
│
├── evaluation/ # Evaluation metrics and agreement analysis
│
├── notebooks/ # End-to-end demo and evaluation notebooks
│
├── requirements.txt # Python dependencies
└── README.md

---

## Data Availability

This repository includes **synthetic demonstration data** intended solely to
illustrate the retrieval, summarization, uncertainty, and evaluation workflows.

Due to copyright restrictions, full-text source articles used in the study are
not redistributed. No copyrighted full-text articles, proprietary databases, or
web-scraped content are included.

The demonstration files mirror the structure and annotation schema of the
original studies but do not constitute a complete dataset and should not be used
for downstream statistical inference.

---

## Reproducibility

All components are implemented using standard Python libraries and deterministic
procedures. The provided notebooks allow users to:

- Run the retrieval and extractive summarization pipeline on example queries  
- Inspect evidence traceability and uncertainty representation  
- Reproduce the evaluation metrics reported in the accompanying manuscript  

The modular design supports transparent inspection, replication, and future
methodological extension.

---

## Intended Use

This codebase is intended for **research and educational purposes only**.

It is not designed to replace expert judgment or to support clinical,
regulatory, or public health decision-making. The framework should not be
interpreted as an automated risk assessment or policy recommendation system.

---

## Citation

If you use this code in your research, please cite the accompanying manuscript:

*Manuscript citation to be added upon publication.*

---

## License

This project is released under the MIT License.
