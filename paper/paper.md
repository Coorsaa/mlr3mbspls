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

`PipeOpMBsPLS` implements sequential orthogonal MB‑sPLS with block-wise sparsity constraints. For each latent component, the method learns one sparse weight vector per block using covariance-style block updates with L1 and L2 constraints. Mean absolute correlation of block scores and a criterion based on squared correlations are available for convergence monitoring, tuning, and evaluation. These criteria do not replace the weight update with a correlation-gradient optimizer. Multiple components are extracted sequentially using deflation adapted to multiblock settings [@westerhuis2001deflation].

Sparsity can be specified either through automatically generated hyperparameters (`c_<block>`, convenient for tuning) or through an explicit sparsity schedule matrix (`c_matrix`, rows = blocks, columns = components) for fully reproducible configurations. The operator outputs per-block latent score columns with consistent names (`LVk_<block>`), enabling downstream clustering, classification, or regression learners to operate directly on the MB‑sPLS representation. For model assessment under resampling, the operator records weights and loadings, as well as block-wise and component-wise explained variance on both training and new data, and the evaluated association criterion.

## Bootstrap stability selection and stable prediction weights

Sparse latent models can be sensitive to sampling variability, especially when blocks are high-dimensional. `PipeOpMBsPLSBootstrapSelect` performs post-hoc bootstrap analysis of MB‑sPLS weights (Efron & Tibshirani [@efron1994bootstrap]) and selects features and components using either (i) confidence intervals (retain features whose bootstrap confidence interval excludes zero) or (ii) selection frequency (retain features whose non-zero frequency exceeds a user-defined threshold), relating to stability-selection ideas [@meinshausen2010stability]. Because MB‑sPLS components are only identifiable up to permutation and sign, bootstrap solutions are aligned prior to aggregation using either block-wise sign rules or score-correlation alignment.

After selection, the operator recomputes stable latent scores using deflation and replaces upstream latent-score columns with the stability-filtered representation, dropping unstable components and blocks. Stable weights can optionally be taken from the original training solution (“training”) or from aligned bootstrap means (“bootstrap_mean”), while selection itself is driven by bootstrap summaries. These stable weight variants are stored so that the upstream MB‑sPLS operator can optionally use stability-filtered weights at prediction time, enabling downstream learners to operate on stable representations without changing the pipeline structure.

## Resampling-friendly validation and diagnostics

To describe whether latent associations generalize, `PipeOpMBsPLS` records held-out objectives and explained variance. Optional fixed-preprocessing or fixed-weight permutation calculations are conditional diagnostics, while the ordinary bootstrap reports descriptive uncertainty and stability rather than a null-hypothesis p-value. A bootstrap test would instead require resampling from a defensible null-constrained model [@hall1991bootstrap].

For confirmatory omnibus inference, `mb_permutation_test()` permutes raw aligned blocks and reruns a complete user-supplied analysis, while `mbspls_permutation_test()` refits a pre-specified MB-sPLS statistic on every shuffle. Both preserve explicit row, stratum, or complete-unit exchangeability and calculate sampled p-values with inclusive ties and the finite Monte Carlo correction [@phipson2010permutation]. The MB-sPLS wrapper deliberately returns one global block- or target-association p-value. It does not present later-component statistics as generic rank tests: simple permutation is known to be invalid for later canonical correlations [@winkler2020cca], and recent PLS simulations found serious limitations in conventional rotated and unrotated per-LV tests [@danyluik2025pls]. Together with `mlr3` measures, explicit group safeguards, and nested cross-validation utilities, these tools support transparent comparison of preprocessing choices, sparsity schedules, and component counts without presenting a partial-pipeline calculation as universal inference.

An orthogonal confirmation design is also supported: `mb_lc_confirmation_test()` tests pre-specified, frozen LC score associations only in observations untouched by fitting or selection, and reports Holm-adjusted p-values across the supplied LC family. This follows the broader sparse-PLS use of hold-out projection to evaluate learned associative effects [@monteiro2016holdout], but the resulting claim is deliberately narrower than population-rank inference: it establishes replication of a fixed score association under the stated exchangeability design.

The package is research software. It cannot infer the correct participant, family, site, scanner, or temporal grouping; causal treatment-effect estimand; confounder set; deployment population; or clinical decision threshold from a dataset. Those study-specific requirements are documented in an installed statistical-validity contract and model-card checklist.

# Acknowledgements

We thank the `mlr3` community for discussions and infrastructure that enabled a pipeline-oriented implementation of multi-block representation learning in R.

# References
