---
title: 'mlr3mbspls: Multi-block sparse PLS representation learning and bootstrap stability selection for mlr3'
tags:
  - R
  - multiblock data
  - sparse partial least squares
  - stability selection
  - multi-omics
  - machine learning pipelines
  - representation learning
authors:
  - name: Stefan Coors
    orcid: 0000-0002-7465-2146
    equal-contrib: true
    affiliation: "1, 3"
  - name: Clara Sophie Vetter
    orcid: 0000-0003-4268-2890
    equal-contrib: true
    affiliation: "2, 3"
affiliations:
  - name: Statistical Learning and Data Science, Ludwig-Maximilians-Universität München, Munich, Germany
    index: 1
  - name: Department of Psychiatry and Psychotherapy, Ludwig-Maximilians-Universität München, Munich, Germany
    index: 2
  - name: Munich Center for Machine Learning (MCML), Munich, Germany
    index: 3
date: 29 December 2025
bibliography: paper.bib
---

# Summary

Multi-block datasets measure multiple feature sets (“blocks”) for the same samples -- for example multi-omics assays, multimodal neuroimaging, or combined clinical and biomarker covariates. Analyses in this setting often require (i) a low-dimensional representation that integrates information across blocks, (ii) block-wise feature selection for interpretability, and (iii) evaluation procedures that remain valid under resampling and hyperparameter tuning.

`mlr3mbspls` provides a native implementation of multi-block sparse partial least squares (MB‑sPLS) as composable preprocessing operators within the `mlr3` machine learning ecosystem [@mlr3] and its pipeline package `mlr3pipelines` [@mlr3pipelines]. MB‑sPLS latent variables are produced as ordinary task features and can therefore be tuned, resampled, and combined with arbitrary downstream learners. The package also includes a bootstrap operator for stability-oriented feature and component selection, which yields stability-filtered latent representations for downstream modeling, and permutation tests that respect an explicit exchangeability design. Core computations are implemented in C++ via Rcpp and RcppArmadillo [@rcpp; @rcpparmadillo] to support high-dimensional blocks and repeated resampling. Source code and usage documentation are available in the public repository [@mlr3mbspls].

# Statement of need

PLS-based integration is widely used for exploratory analysis and prediction with heterogeneous, high-dimensional biomedical data. In R, `mixOmics` offers a comprehensive toolbox for (sparse) PLS-based multi-omics integration and visualization [@mixomics], `multiblock` collects a broad range of multiblock methods behind common interfaces [@multiblock], and sparse PLS formulations have been studied extensively for variable selection in high-dimensional problems [@lecao2008spls]. However, these workflows are commonly implemented as bespoke scripts. This makes it harder to (a) benchmark competing preprocessing choices in a leakage-free way, (b) tune sparsity schedules and component counts under cross-validation, and (c) couple representation learning with downstream learners using a consistent interface.

`mlr3mbspls` addresses this gap by turning MB‑sPLS into first-class, trainable transformers in `mlr3pipelines`. This design allows users to place MB‑sPLS within directed acyclic graphs that also include imputation, scaling, batch/site correction, and downstream learners, while preserving the separation of training and prediction phases required for valid model evaluation. In addition, the bootstrap selection operator quantifies how sparse weight vectors vary under resampling and restricts the representation to consistently selected features. The target audience are applied researchers and method developers who work with multi-block data and want an interoperable workflow for representation learning, feature selection, and downstream modeling in the `mlr3` ecosystem.

# Methods and implementation

## MB‑sPLS as an `mlr3pipelines` transformer

`PipeOpMBsPLS` implements sequential MB‑sPLS with block-wise sparsity constraints. For each component, one sparse weight vector per block is learned by block-coordinate penalized-matrix-decomposition updates [@witten2009pmd] towards the mean standardized score of the other blocks, under a unit L2 norm and an L1 budget. The updates start deterministically, so fits do not depend on the random seed; because the criterion is non-convex, the result is a local optimum rather than a guaranteed global maximum. The mean absolute correlation of block scores, or a squared-correlation criterion, is used for tuning and evaluation. Later components follow block-wise score deflation [@westerhuis2001deflation].

Sparsity can be specified either through automatically generated hyperparameters (`c_<block>`, convenient for tuning) or through an explicit sparsity schedule matrix (`c_matrix`, rows = blocks, columns = components) for fully reproducible configurations. The operator outputs per-block latent score columns (`LVk_<block>`), computed from the training fit at both training and prediction time, so that downstream clustering, classification, or regression learners can operate directly on the MB‑sPLS representation. For model assessment under resampling, it records weights, loadings, explained variance on training and new data, and the evaluated association criterion.

## Bootstrap stability selection

Sparse latent models can be sensitive to sampling variability, especially when blocks are high-dimensional. `PipeOpMBsPLSBootstrapSelect` refits MB‑sPLS on bootstrap resamples of the training data [@efron1994bootstrap], resampling whole groups when the task defines a grouping. Replicate components are matched to the training components and sign-aligned block by block. Features are selected by percentile intervals that exclude zero together with a minimum absolute mean weight, or by a non-zero selection frequency of at least a threshold, relating to stability-selection ideas [@meinshausen2010stability]; the intervals are stability summaries, not confidence intervals for population weights.

After selection, the operator recomputes stable latent scores with deflation and replaces the upstream latent-score columns consistently at training and prediction time, dropping components and blocks without stable features. Stable weights are either the training weights restricted to the selected features or aligned bootstrap means; a stability-only mode stores the summaries without changing the task. Because selection is part of the pipeline, nested resampling evaluates it together with tuning and fitting.

## Resampling-friendly validation and diagnostics

To describe whether latent associations generalize, `PipeOpMBsPLS` records held-out objectives and explained variance. Fixed-preprocessing or fixed-weight permutation calculations are labelled as conditional diagnostics, and the tuners' early-stopping rules as optimistic heuristics. The ordinary bootstrap reports descriptive uncertainty rather than a p-value; a bootstrap test would require resampling from a null-constrained model [@hall1991bootstrap].

For confirmatory omnibus inference, `mb_permutation_test()` permutes raw aligned blocks and reruns a complete user-supplied analysis, while `mbspls_permutation_test()` refits a pre-specified MB-sPLS statistic on every shuffle. Both preserve explicit row, stratum, or complete-unit exchangeability, calculate sampled p-values with inclusive ties and the finite Monte Carlo correction [@phipson2010permutation], and report Monte Carlo precision as an exact binomial interval [@clopper1934binomial]. The MB-sPLS wrapper deliberately returns one global block- or target-association p-value. It does not present later-component statistics as generic rank tests: simple permutation is known to be invalid for later canonical correlations [@winkler2020cca], and recent PLS simulations found serious limitations in conventional rotated and unrotated per-LV tests [@danyluik2025pls].

An orthogonal confirmation design is also supported: `mb_lc_confirmation_test()` tests pre-specified, frozen LC score associations only in observations untouched by fitting or selection, and reports Holm-adjusted p-values [@holm1979] across the supplied LC family. This follows the sparse-PLS use of hold-out projection to evaluate learned associative effects [@monteiro2016holdout], with a deliberately narrower claim than population-rank inference. Given the signs of the pairwise score correlations from discovery, each test is one-sided and supports replication in the discovery direction on average across block pairs; without them it establishes dependence in either direction.

The package is research software. It cannot infer the correct participant, family, site, scanner, or temporal grouping; causal treatment-effect estimand; confounder set; deployment population; or clinical decision threshold from a dataset. Those study-specific requirements are documented in an installed statistical-validity contract and model-card checklist.

# Acknowledgements

We thank the `mlr3` community for discussions and infrastructure that enabled a pipeline-oriented implementation of multi-block representation learning in R.

# References
