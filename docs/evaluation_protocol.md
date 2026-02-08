## Evaluation Protocol

This document describes the evaluation protocol used to assess the proposed interpretable, evidence-centered environmental health question-answering framework.

The protocol is designed to evaluate not only evidence retrieval behavior, but also factual consistency, interpretability, citation traceability, and qualitative uncertainty alignment, consistent with the methodology reported in the accompanying manuscript.

## Evaluation Objectives

The evaluation addresses the following questions:

Does the system retrieve literature evidence relevant to a given pollutant–health query?

Are the generated summaries factually consistent with the cited evidence?

Are system outputs interpretable and transparent to domain experts?

Do system-assigned qualitative confidence levels align with expert judgment?

Is expert evaluation reasonably consistent across reviewers?

## Evaluation Scenarios

A set of representative pollutant–health query scenarios is defined to span diverse exposure types and health outcomes.

Each scenario consists of:

A natural-language query

A target pollutant

A target health outcome

In this repository, synthetic demonstration scenarios are provided solely to illustrate the evaluation workflow and metric computation. These scenarios do not represent the full evaluation dataset reported in the manuscript.

## Evidence Retrieval Evaluation
Citation Precision@k

Retrieval performance is assessed using citation precision@k, defined as:

The proportion of expert-judged relevant studies among the top-k retrieved documents.

For each query scenario:

Retrieved documents are independently labeled by experts as relevant (1) or not relevant (0).

Precision@k is computed as the mean of relevance labels among the top-k documents.

In the reported study, k = 3, reflecting a balance between evidence diversity and expert inspectability.

## Expert Review Process
Expert Panel

System outputs are independently reviewed by ten external domain experts with complementary expertise spanning environmental health, public health, and related disciplines.

Experts:

Are external to the author team

Declare no conflicts of interest

Review all outputs independently

Are blinded to internal retrieval scores and uncertainty calibration rules

## Evaluation Dimensions

Experts assess each system output along the following dimensions:

1. Factual Consistency

Three-level ordinal scale:

0 = incorrect or unsupported by the cited evidence

1 = broadly consistent but incomplete or missing key qualifiers

2 = fully consistent with the cited evidence

This scale reflects graded factual alignment rather than binary correctness.

2. Interpretability

Rated on a 5-point Likert scale

Assesses clarity, transparency, and ease of inspection, including:

Evidence presentation

Citation traceability

Confidence labeling

3. Expert-Perceived Confidence

Ordinal categorical judgment:

Low, Medium, or High

Reflects the expert’s assessment of the strength, consistency, and adequacy of the supporting evidence.

## Uncertainty Alignment
## System Confidence Assignment

System-assigned qualitative confidence levels are derived from retrieval-based evidence support signals, following the procedure described in the manuscript:

Document-level similarity scores from the top-k retrieved studies are normalized using min–max normalization.

Normalized scores are aggregated using their arithmetic mean.

Two predefined thresholds (fixed prior to evaluation) map the aggregated score to:

Low

Medium

High confidence

This procedure is deterministic and reproducible and is intended as an interpretable proxy for relative evidence support rather than probabilistic calibration.

## Expert Confidence Aggregation

Expert-perceived confidence labels are aggregated per query using majority voting.

The resulting label represents expert consensus confidence.

## Alignment Metric

Alignment between system-assigned and expert-perceived confidence is assessed using weighted Cohen’s κ with linear weights, reflecting ordinal agreement.

κ is interpreted as a measure of interpretability alignment, not predictive accuracy or statistical calibration.

## Inter-Rater Agreement

Consistency among expert reviewers is summarized descriptively.

Inter-rater agreement is assessed using Cohen’s κ for selected evaluation dimensions.

Agreement statistics are reported to contextualize expert judgment variability rather than to support inferential claims.

## Statistical Reporting

Evaluation outcomes are summarized using descriptive statistics:

Mean ± standard deviation for ordinal or continuous measures

Proportions for categorical measures

No hypothesis testing or inferential statistical claims are made, consistent with the proof-of-concept scope.

## Demonstration Data and Reproducibility

The repository includes synthetic demonstration data (CSV format) to illustrate:

Metric computation

Evaluation logic

Data schema

These data:

Do not represent real expert judgments

Do not contain real study outputs

Are not used for any results reported in the manuscript

The schema supports direct substitution with real evaluation data.

## Scope and Limitations

This evaluation protocol is designed for methodological demonstration rather than generalizable performance benchmarking.

Limitations include:

A limited number of demonstration scenarios

Use of synthetic data in the repository

Focus on expert-based evaluation rather than end-user usability testing

These limitations are discussed in detail in the manuscript and motivate future extensions.

## Intended Interpretation

Evaluation results should be interpreted as evidence of methodological feasibility, transparency, and interpretability, rather than as indicators of predictive accuracy or decision-support validity.

The framework is intended to support, not replace, expert judgment in environmental health research.