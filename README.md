Interpretable Environmental Health Question-Answering Framework

This repository provides a proof-of-concept implementation of an interpretable, evidence-based question-answering framework for environmental health research.
The system is designed to support transparent synthesis of pollutant–health evidence from scientific literature, with explicit uncertainty representation and full citation traceability.

The code accompanies an academic manuscript entitled
“Development and Evaluation of a Transparent AI Framework for Evidence-Centered Question Answering in Environmental Health Research.”

Overview

Environmental health evidence is often dispersed across heterogeneous studies and presented in formats that are difficult for non-specialists to interpret.
This framework explores how AI-inspired retrieval and reasoning techniques can be combined with structured knowledge representation to support:

Natural-language question answering on pollutant–health relationships

Evidence-grounded answer generation

Explicit uncertainty calibration

Transparent linkage between answers and source literature

Rather than relying on black-box large language model outputs, the system emphasizes interpretability, reproducibility, and traceability as first-class design principles.

This repository implements a lightweight, non-generative prototype intended to demonstrate methodological feasibility rather than a production-level decision-support system.

Framework Components

The repository is organized to reflect the main methodological components of the framework:

Evidence Retrieval

TF–IDF vectorization and cosine similarity are used to retrieve relevant literature snippets in response to a query.

Extractive Summarization

Key sentences exhibiting lexical overlap with query terms are selected to form concise, evidence-based summaries.

Uncertainty Calibration

Retrieval similarity scores are transformed into qualitative confidence levels (Low / Medium / High) using a sigmoid-based heuristic.

Evaluation

Scripts are provided to compute citation precision, inter-rater agreement (Cohen’s κ), and interpretability-related statistics as reported in the accompanying manuscript (Table 1).

Repository Structure
env-health-qa-framework/
│
├── data/
│   └── demo_corpus/        # Synthetic demonstration corpus (not full literature set)
│
├── retrieval/              # TF–IDF indexing and similarity-based search
│
├── summarization/          # Extractive summarization methods
│
├── uncertainty/            # Uncertainty calibration utilities
│
├── evaluation/             # Evaluation metrics and statistical scripts
│
├── notebooks/              # Jupyter notebooks demonstrating the full pipeline
│
├── docs/                   # Additional methodological documentation
│
├── requirements.txt        # Python dependencies
└── README.md

Data Availability

This repository includes a synthetic demonstration corpus intended solely to illustrate the retrieval, reasoning, and evaluation pipeline.

Due to copyright restrictions, full-text source articles used in the study are not redistributed.
No copyrighted full-text articles, proprietary databases, or web-scraped content are included in this repository.
The demonstration files mirror the structure and annotations of the original studies but do not constitute a complete dataset and should not be used for downstream statistical inference.

Reproducibility

The provided scripts and notebooks allow users to:

Run the retrieval and extractive summarization pipeline on example inputs

Reproduce the qualitative uncertainty calibration behavior

Replicate the evaluation metrics reported in the accompanying manuscript

All components are implemented using standard Python libraries and are organized to support transparent inspection, replication, and extension of the proof-of-concept workflow.

Intended Use

This codebase is intended for research and educational purposes only.
It is not designed to replace expert judgment or to support clinical, regulatory, or public health decision-making.

The framework is provided as a methodological demonstration and should not be interpreted as an automated risk assessment or policy recommendation system.

Citation

If you use this code in your research, please cite the accompanying manuscript:

Manuscript citation to be added upon publication.

License

This project is released under the MIT License.