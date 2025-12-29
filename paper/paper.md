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

`mlr3mbspls` provides a native implementation of multi-block sparse partial least squares (MB‑sPLS) as composable preprocessing operators within the `mlr3` machine learning ecosystem [@mlr3] and its pipeline package `mlr3pipelines` [@mlr3pipelines]. MB‑sPLS latent variables are produced as ordinary task features and can therefore be tuned, resampled, and combined with arbitrary downstream learners. The package also includes a dedicated bootstrap operator that performs stability-oriented feature and component selection, yielding stable weight vectors and stability-filtered latent representations for downstream modeling. Core computations are implemented in C++ via Rcpp and RcppArmadillo [@rcpp; @rcpparmadillo] to support high-dimensional blocks and repeated resampling. Source code and usage documentation are available in the public repository [@mlr3mbspls].

# Statement of need

PLS-based integration is widely used for exploratory analysis and prediction with heterogeneous, high-dimensional biomedical data. In R, `mixOmics` offers a comprehensive toolbox for (sparse) PLS-based multi-omics integration and visualization [@mixomics], and sparse PLS formulations have been studied extensively for variable selection in high-dimensional problems [@lecao2008spls]. However, these workflows are commonly implemented as bespoke scripts. This makes it harder to (a) benchmark competing preprocessing choices in a leakage-free way, (b) tune sparsity schedules and component counts under cross-validation, and (c) couple representation learning with downstream learners using a consistent interface.

`mlr3mbspls` addresses this gap by turning MB‑sPLS into first-class, trainable transformers in `mlr3pipelines`. This design allows users to place MB‑sPLS within directed acyclic graphs that also include imputation, scaling, batch/site correction, and downstream learners, while preserving the separation of training and prediction phases required for valid model evaluation. In addition, the bootstrap selection operator supports stability-oriented model interpretation by reducing the sensitivity of sparse weight vectors to sampling variation. The target audience are applied researchers and method developers who work with multi-block data and want an interoperable workflow for representation learning, feature selection, and downstream modeling in the `mlr3` ecosystem.

# Methods and implementation

## MB‑sPLS as an `mlr3pipelines` transformer

`PipeOpMBsPLS` implements sequential orthogonal MB‑sPLS with block-wise sparsity constraints. For each latent component, the method learns one sparse weight vector per block and optimizes a global association criterion across blocks. The package supports two objectives: mean absolute correlation of block scores (a scale-free association measure) and a Frobenius-norm objective based on squared correlations. Multiple components are extracted sequentially using deflation adapted to multiblock settings [@westerhuis2001deflation].

Sparsity can be specified either through automatically generated hyperparameters (`c_<block>`, convenient for tuning) or through an explicit sparsity schedule matrix (`c_matrix`, rows = blocks, columns = components) for fully reproducible configurations. The operator outputs per-block latent score columns with consistent names (`LVk_<block>`), enabling downstream clustering, classification, or regression learners to operate directly on the MB‑sPLS representation. For model assessment under resampling, the operator records weights and loadings, as well as block-wise and component-wise explained variance on both training and new data, and the optimized association objective.

## Bootstrap stability selection and stable prediction weights

Sparse latent models can be sensitive to sampling variability, especially when blocks are high-dimensional. `PipeOpMBsPLSBootstrapSelect` performs post-hoc bootstrap analysis of MB‑sPLS weights (Efron & Tibshirani [@efron1994bootstrap]) and selects features and components using either (i) confidence intervals (retain features whose bootstrap confidence interval excludes zero) or (ii) selection frequency (retain features whose non-zero frequency exceeds a user-defined threshold), relating to stability-selection ideas [@meinshausen2010stability]. Because MB‑sPLS components are only identifiable up to permutation and sign, bootstrap solutions are aligned prior to aggregation using either block-wise sign rules or score-correlation alignment.

After selection, the operator recomputes stable latent scores using deflation and replaces upstream latent-score columns with the stability-filtered representation, dropping unstable components and blocks. Stable weights can optionally be taken from the original training solution (“training”) or from aligned bootstrap means (“bootstrap_mean”), while selection itself is driven by bootstrap summaries. These stable weight variants are stored so that the upstream MB‑sPLS operator can optionally use stability-filtered weights at prediction time, enabling downstream learners to operate on stable representations without changing the pipeline structure.

## Resampling-friendly validation and diagnostics

To quantify whether latent associations generalize, `PipeOpMBsPLS` includes optional train-time permutation testing (as an early-stopping criterion for component extraction) and prediction-side validation of the latent association objective via permutation or bootstrap inference. Together with `mlr3` measures for MB‑sPLS objectives and explained variance, these tools support benchmarking and nested cross-validation workflows that compare preprocessing choices, sparsity schedules, and component counts.

# Acknowledgements

We thank the `mlr3` community for discussions and infrastructure that enabled a pipeline-oriented implementation of multi-block representation learning in R.

# References
