---
title: 'mlr3mbspls: Multi-block sparse PLS transformers and stability selection for multimodal learning in mlr3'
tags:
  - R
  - machine learning
  - multiblock
  - multi-omics
  - partial least squares
  - feature selection
  - clustering
authors:
  - name: Stefan Coors
    affiliation: 1
    corresponding: true
  - name: Clara Sophie Vetter
    affiliation: 1
affiliations:
  - name: Ludwig-Maximilians-Universität München, Germany
    index: 1
date: 29 December 2025
bibliography: paper.bib
---

# Summary

Modern biomedical and industrial studies increasingly combine multiple data modalities measured on the same samples (e.g., clinical covariates, transcriptomics, proteomics, metabolomics, or imaging-derived features). Multi-block partial least squares (MB-PLS) models are a natural fit for these settings because they learn *block-wise* latent variables that capture shared structure across modalities, while remaining interpretable at the level of the original features.

`mlr3mbspls` is an R package that implements **multi-block sparse PLS (MB-sPLS)** as first-class operators in the `mlr3` ecosystem [@mlr3; @mlr3pipelines]. The package provides pipeline building blocks (`PipeOp`s) for (i) learning multi-block latent representations with block-wise feature sparsity and (ii) performing bootstrap-based stability selection to obtain more robust, reproducible feature sets and latent variables.

The source code is developed openly in a public Git repository [@mlr3mbspls].

# Statement of need

In practice, multi-block latent variable methods are rarely used in isolation: they must be combined with preprocessing (scaling, batch/site correction), resampling, hyperparameter tuning, and downstream learners (clustering, classification, regression). While established R toolkits such as `mixOmics` [@rohart2017mixomics; @singh2019diablo] and `multiblock` [@multiblock] provide rich implementations of multiblock PLS variants, integrating these methods into *end-to-end* machine learning workflows—without leakage between training and test splits, and with consistent logging of intermediate quantities—requires substantial bespoke code.

`mlr3mbspls` addresses this gap by exposing MB-sPLS as composable pipeline operators that work natively with `mlr3` resampling and tuning. In addition, the package implements bootstrap stability selection within the same pipeline abstraction, enabling stable feature selection and stable latent variables that can be propagated to prediction and downstream learners.

# MB-sPLS as an mlr3 pipeline operator

The core transformer is `PipeOpMBsPLS`. Given a user-defined *block map* from modalities to feature columns, the operator learns a set of block scores (latent variables) for each component and returns them as new features (e.g., `LV1_genomics`, `LV1_proteomics`). The implementation follows a sequential multiblock PLS formulation with deflation, allowing components to capture complementary shared structure across blocks [@westerhuis2001deflation].

High-dimensional settings motivate sparsity in the loading weights to improve interpretability and reduce overfitting. `PipeOpMBsPLS` supports **block-specific and component-specific sparsity** via either (i) per-block sparsity parameters or (ii) an explicit sparsity matrix, generalizing sparse PLS ideas that use ℓ1-type penalties or soft-thresholding to induce sparse loading vectors [@lecao2008spls]. Computationally intensive steps are implemented in C++ via `RcppArmadillo`.

Beyond representation learning, the operator records diagnostic quantities (objective trajectories, explained variance by block and component, weight sparsity) and supports optional **permutation-based component stopping**. At prediction time, `PipeOpMBsPLS` can optionally perform out-of-sample permutation or bootstrap tests for latent correlations, providing an additional sanity check on whether the learned cross-block signal generalizes.

# Bootstrap stability selection for MB-sPLS

Sparse multiblock models can be sensitive to sampling noise: different training splits may yield different selected features and component orientations. To improve robustness, `mlr3mbspls` provides `PipeOpMBsPLSBootstrapSelect`, a downstream operator that runs MB-sPLS on multiple bootstrap resamples [@efron1994bootstrap] and derives stability statistics for each feature and component.

The operator (i) aligns components across bootstrap replicates (to resolve sign and ordering ambiguities), (ii) aggregates loading weights into bootstrap means and confidence intervals, and (iii) selects stable features either by **confidence-interval filtering** (interval excludes zero) or by **selection frequency** thresholds. This workflow is closely related in spirit to stability selection ideas that emphasize reproducibility of selected variables under resampling [@meinshausen2010stability].

After selecting stable weights, the operator recomputes **stable latent variables** by projecting the original blocks onto the selected weight vectors and applying component-wise deflation. The stable weights can be stored and reused by `PipeOpMBsPLS` for predicting new data, enabling a consistent latent representation between training and deployment.

# Usage in end-to-end mlr3 workflows

`mlr3mbspls` integrates with `mlr3pipelines` graphs so that preprocessing, MB-sPLS, stability selection, and downstream learners can be composed and tuned within nested resampling. For example, a typical unsupervised pipeline for multimodal clustering can be built as:

```r
library(mlr3)
library(mlr3cluster)
library(mlr3pipelines)
library(mlr3mbspls)

blocks = list(
  genomics   = grep("^rna_", task$feature_names, value = TRUE),
  proteomics = grep("^prot_", task$feature_names, value = TRUE)
)

glrn = mbspls_graph_learner(
  learner = lrn("clust.kmeans", centers = 3L),
  blocks = blocks,
  ncomp = 3L,
  bootstrap_selection = TRUE,
  B = 200L,
  selection_method = "ci"
)

rr = resample(task, glrn, rsmp("cv", folds = 5))
```

The same graph can be tuned with `mlr3tuning` to optimize sparsity levels and the number of components using nested cross-validation, while `autoplot()` methods provide visualizations of weights, cross-block correlations, explained variance, and bootstrap stability diagnostics.

# References
