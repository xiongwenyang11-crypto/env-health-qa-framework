# Method Overview

This document provides a high-level overview of the design rationale and methodological structure of the interpretable environmental health question-answering framework implemented in this repository.

The goal of the framework is to support transparent, evidence-based synthesis of environmental health literature, with explicit uncertainty representation and full traceability to source evidence.

---

## Design Principles

The framework is guided by the following core principles:

1. **Evidence-driven reasoning**  
   All generated answers are grounded in retrieved literature evidence rather than free-form text generation.

2. **Interpretability over black-box prediction**  
   Each stage of the pipeline is explicitly defined and inspectable, avoiding opaque end-to-end models.

3. **Traceability and transparency**  
   Users can trace system outputs back to specific studies and text snippets.

4. **Explicit uncertainty representation**  
   Confidence levels are treated as first-class outputs rather than implicit by-products.

5. **Reproducibility**  
   The system is implemented using standard, widely available Python libraries and includes demonstration datasets to support reproducibility.

---

## System Architecture

The framework follows a modular pipeline architecture consisting of four main components:

1. Evidence Retrieval  
2. Extractive Summarization  
3. Uncertainty Calibration  
4. Evaluation and Validation  

Each component is implemented as an independent module to support clarity, extensibility, and reproducibility.

---

## Evidence Retrieval

The retrieval module identifies literature relevant to a user’s natural-language query.

- Texts are indexed using a TF–IDF vector representation.
- Cosine similarity is used to rank documents by relevance.
- The top-*k* documents are returned along with similarity scores.

This design prioritizes interpretability and reproducibility over complex neural retrieval models, which may obscure how relevance scores are produced.

For demonstration purposes, retrieval is performed at the document level. The architecture can be extended to sentence- or paragraph-level retrieval in future work.

---

## Extractive Summarization

Retrieved documents are summarized using a query-aware extractive approach:

- Documents are split into sentences using lightweight rule-based segmentation.
- Sentences are scored based on lexical overlap with query terms.
- Top-ranked sentences across retrieved documents are selected to form the final summary.

This extractive strategy ensures that summaries remain faithful to the source texts and supports direct traceability between the generated answer and the underlying evidence.

---

## Uncertainty Calibration

To represent uncertainty explicitly, retrieval similarity scores are transformed into qualitative confidence levels.

The calibration process includes:
1. Optional min–max normalization of similarity scores.
2. Sigmoid-based transformation to smooth score distributions.
3. Threshold-based mapping into three qualitative levels:
   - **Low**
   - **Medium**
   - **High**

Each retrieved evidence item receives a confidence label, and an overall confidence level is derived for the final answer (e.g., based on the top-ranked evidence).

This approach provides an interpretable approximation of uncertainty aligned with expert judgment, rather than probabilistic claims.

---

## Evaluation Strategy

The evaluation framework is designed to assess both factual correctness and interpretability.

Key evaluation dimensions include:

- **Citation precision@k**  
  Proportion of relevant studies among the top-*k* retrieved items.

- **Factuality**  
  Binary expert judgment of whether the generated summary is consistent with the cited evidence.

- **Interpretability**  
  Expert-rated clarity and transparency of system outputs using a Likert scale.

- **Uncertainty alignment**  
  Agreement between system-assigned confidence levels and expert-perceived confidence.

- **Inter-rater agreement**  
  Pairwise Cohen’s κ is used to quantify consistency among expert reviewers.

Synthetic demonstration data are used in this repository to illustrate the evaluation workflow. Real expert ratings and literature data are not redistributed.

---

## Demonstration Data vs. Real Literature

This repository includes small, synthetic demonstration datasets intended solely to illustrate the pipeline and evaluation procedures.

- The **demo corpus** does not represent the full literature used in the study.
- Texts are simplified or synthetic to avoid copyright restrictions.
- Evaluation CSV files contain synthetic expert ratings for reproducibility only.

The structure of the code and data schemas is designed to support seamless substitution with real literature and expert evaluations in future work.

---

## Scope and Extensions

This implementation represents a proof-of-concept framework rather than a production system.

Planned extensions include:
- Large-scale literature databases
- Embedding-based retrieval
- Automated entity recognition
- Integration with more advanced language model reasoning modules

These extensions are intentionally beyond the scope of the current implementation and manuscript.

---

## Intended Use

This framework and accompanying code are intended for research and educational purposes only.

They are not designed to replace expert judgment or to support clinical or public health decision-making.
