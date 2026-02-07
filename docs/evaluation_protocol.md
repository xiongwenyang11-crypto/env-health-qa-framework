# Evaluation Protocol

This document describes the evaluation protocol used to assess the proposed interpretable environmental health question-answering framework.

The protocol is designed to evaluate not only retrieval performance, but also factual correctness, interpretability, and the appropriateness of uncertainty representation.

---

## Evaluation Objectives

The evaluation aims to answer the following questions:

1. Does the system retrieve relevant literature evidence for a given query?
2. Are the generated summaries factually consistent with the cited evidence?
3. Are system outputs interpretable and transparent to expert users?
4. Do system-assigned confidence levels align with expert judgment?
5. Is there reasonable agreement among expert reviewers?

---

## Evaluation Scenarios

A set of representative pollutant–health query scenarios is defined to cover diverse exposure types and health outcomes.

Each scenario consists of:
- A natural-language query
- A target pollutant
- A target health outcome

In this repository, a small number of **synthetic demonstration scenarios** are provided to illustrate the evaluation workflow.

---

## Evidence Retrieval Evaluation

### Citation Precision@k

Retrieval performance is assessed using citation precision@k, defined as:

> The proportion of relevant studies among the top-*k* retrieved items.

For each scenario:
- Retrieved evidence items are labeled as relevant (1) or not relevant (0) by expert reviewers.
- Precision@k is computed as the mean of these relevance labels.

This metric reflects the system’s ability to prioritize relevant literature for downstream reasoning and summarization.

---

## Expert Review Process

### Expert Panel

System outputs are independently reviewed by three domain experts with expertise in environmental health research.

Experts are blinded to each other’s assessments and evaluate system outputs independently.

---

### Evaluation Dimensions

Experts assess each system output across the following dimensions:

#### 1. Factuality

- Binary judgment (0 / 1)
- Indicates whether the generated summary is factually consistent with the cited evidence.

#### 2. Interpretability

- Rated on a 5-point Likert scale
- Reflects clarity, transparency, and ease of understanding of system outputs, including:
  - Evidence presentation
  - Confidence labeling
  - Traceability to source studies

#### 3. Perceived Confidence

- Categorical judgment: **Low**, **Medium**, or **High**
- Represents the expert’s perception of evidence strength and consistency.

---

## Uncertainty Alignment

System-assigned confidence levels are compared with expert-perceived confidence levels.

### System Confidence Assignment

- Retrieval similarity scores are calibrated using a sigmoid-based transformation.
- Scores are mapped to qualitative levels:
  - Low
  - Medium
  - High

An overall system confidence level is derived per scenario, typically based on the top-ranked evidence.

---

### Expert Confidence Aggregation

- For each scenario, expert confidence labels are aggregated using majority vote.
- This aggregated label represents the expert consensus confidence level.

---

### Alignment Metric

Uncertainty alignment is defined as:

> Exact agreement between the system-assigned confidence label and the expert majority confidence label.

Alignment is reported as:
- Per-scenario agreement (0 / 1)
- Overall alignment rate across scenarios

---

## Inter-Rater Agreement

To assess consistency among expert reviewers, inter-rater agreement is quantified using **pairwise Cohen’s κ**.

- Cohen’s κ is computed for each pair of experts.
- Mean κ values are reported to summarize agreement.

Interpretation follows standard guidelines:
- κ < 0.40: low agreement
- 0.40 ≤ κ < 0.60: moderate agreement
- 0.60 ≤ κ < 0.80: substantial agreement
- κ ≥ 0.80: near-perfect agreement

---

## Statistical Reporting

Evaluation results are summarized using:

- Mean ± standard deviation for continuous measures
- Proportions for binary and categorical measures

Non-parametric statistical tests (e.g., Wilcoxon signed-rank test) may be used for exploratory comparisons where appropriate.

---

## Demonstration Data and Reproducibility

The evaluation workflow provided in this repository uses **synthetic demonstration data** stored in CSV format.

- Demonstration data illustrate evaluation logic and metric computation.
- Values do not represent real expert judgments or real study outputs.
- Full literature texts and real expert ratings are not redistributed due to copyright and privacy considerations.

The data schema is designed to support seamless substitution with real evaluation data.

---

## Scope and Limitations

The evaluation protocol is designed for proof-of-concept validation.

Limitations include:
- Small number of scenarios
- Synthetic demonstration data
- Focus on expert-based evaluation rather than end-user usability testing

These limitations are addressed in the accompanying manuscript and motivate future system extensions.

---

## Intended Interpretation

Evaluation results should be interpreted as evidence of **methodological feasibility and transparency**, rather than as definitive performance benchmarks.

The framework is intended to support, not replace, expert judgment in environmental health research.
