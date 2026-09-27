---
title: 'mlr3mbspls: Multi-block sparse partial least squares with bootstrap stability selection and permutation inference for mlr3'
tags:
  - R
  - multi-block data
  - data integration
  - sparse partial least squares
  - bootstrap stability selection
  - permutation inference
  - machine learning pipelines
  - mlr3
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
    ror: "05591te55"
  - name: Department of Psychiatry and Psychotherapy, Ludwig-Maximilians-Universität München, Munich, Germany
    index: 2
    ror: "05591te55"
  - name: Munich Center for Machine Learning (MCML), Munich, Germany
    index: 3
    ror: "02nfy3535"
date: 26 September 2026
bibliography: paper.bib
---

# Summary

Many studies measure several kinds of data on the same participants, for example brain scans, questionnaires and blood markers. Each kind of data forms a "block" of variables. Researchers often want, for each block, a few summary scores that each use a small, interpretable subset of its variables and that vary together across blocks. Multi-block sparse partial least squares (MB-sPLS) is a statistical method that finds such scores.

`mlr3mbspls` is an R package that makes MB-sPLS and related methods steps of machine-learning pipelines in the `mlr3` ecosystem [@mlr3; @mlr3pipelines]. The scores can be combined with data cleaning, correction for measurement sites and prediction or clustering models, and the whole pipeline can be cross-validated without test data leaking into the fit. The package also measures how stable the selected variables are and tests whether associations could have arisen by chance, respecting study structure such as repeated visits.

# Statement of need

Sparse partial least squares (PLS) and related latent-variable methods are widely used to relate high-dimensional data such as omics assays, brain imaging and clinical measures [@lecao2008spls; @singh2019diablo; @mihalik2022ccapls]. Such analyses face three pitfalls. Their estimates are unstable at typical sample sizes [@helmer2024stability]. Performance estimates can be optimistically biased when any data-dependent step, such as imputation, site correction, scaling, sparsity tuning or variable selection, sees the evaluation data [@ambroise2002selection; @varma2006bias; @kapoor2023leakage]. And permutation tests are valid only when the shuffles respect the exchangeability structure of the data, such as participants, families or repeated visits [@winkler2015multilevel].

`mlr3mbspls` addresses these pitfalls within `mlr3pipelines`. Every data-dependent step is an operator refitted in each resampling fold; bootstrap stability selection quantifies instability and can restrict the scores to consistently selected variables; and the confirmatory permutation tests shuffle rows, rows within strata, or whole units such as participants, as the user declares. The operators cover unsupervised MB-sPLS, a supervised variant that treats the outcome as an additional block, multi-block sparse principal component analysis [@witten2009pmd], block scaling and site correction, including ComBat [@johnson2007combat; @fortin2018harmonization]. Each MB-sPLS component has one sparse weight vector per block; the weighted sum of a block's centred variables (for later components, of what earlier components leave unexplained) is its score, which becomes a new feature. The target audience is researchers who analyse multi-block data, for example in psychiatry, neuroimaging or multi-omics, and methodologists comparing such representations with other pipelines.

# State of the field

In R, `mixOmics` provides multi-block sparse PLS with prediction for new samples, and its discriminant form DIABLO adds cross-validated tuning of the number of selected variables [@mixomics; @singh2019diablo]. `RGCCA` implements regularised generalised canonical correlation analysis (CCA) and its sparse variant SGCCA, with cross-validation, permutation-based penalty tuning, bootstrap intervals and stability-based variable selection [@rgcca; @tenenhaus2011rgcca; @tenenhaus2014sgcca]. `PMA` provides sparse multiple CCA [@witten2009mcca]; `multiblock` and `ade4` collect many multi-block methods, including sparse multi-block PLS in `multiblock` [@multiblock; @smilde2022multiblock; @bougeard2018ade4]. In Python, `mbpls` offers non-sparse multi-block PLS [@baum2019mbpls], and `cca-zoo` provides sparse multiview CCA usable as a scikit-learn pipeline step [@chapman2021ccazoo; @pedregosa2011sklearn]. Single-block PLS is available as a tidymodels preprocessing step [@recipes] and as `mlr3` learners [@fischer2025mlr3extralearners], but `mlr3pipelines` has no PLS, CCA or multi-block operator.

The MB-sPLS method implemented here was developed by C. S. Vetter, who first implemented it in MATLAB; it is closely related to SGCCA and sparse multiple CCA. We did not find in R (i) scores, bootstrap selection and site correction as pipeline operators refitted, with sparsity tuning, in every resampling fold, or (ii) permutation inference for multi-block sparse PLS under explicit exchangeability designs. `RGCCA` and `PMA` tune penalties with unrestricted shuffles, and restricted permutations for two-view CCA and PLS exist in a MATLAB toolkit [@mihalik2022ccapls].

We implemented `mlr3mbspls` as a separate `mlr3` extension because we build it on top of `mlr3pipelines` operators. These packages are built around fitting and inspecting a multi-block model, and their validation helpers resample that model alone; `mlr3mbspls` treats MB-sPLS as one resampled step of a larger pipeline. Adding the operator contract to them would import `mlr3`'s class, parameter and tuning infrastructure into packages with different design goals, whereas `mlr3` distributes such integrations as extension packages [@fischer2025mlr3extralearners].

# Software design

**Operators with training state.** Each operator learns its state (such as block columns, centring means, weights and selected variables) from the training rows and applies it unchanged at prediction. A standalone function would be simpler interactively but would leave train--test separation to the user; with operators, `mlr3` resampling, tuning and benchmarking refit every step on each fold's training rows. Splits must still keep dependence units such as participants together, which the tuners and nested cross-validation check for groups declared in the task.

**A shared deterministic solver.** Each component is fitted by block-coordinate updates in the style of the penalised matrix decomposition [@witten2009pmd]: in turn, each block's weight vector, constrained to unit L2 norm and an L1 budget, is set to maximise the covariance between that block's score and the mean standardised score of the other blocks. We implemented it in C++ [@rcpp; @rcpparmadillo] rather than wrapping `RGCCA` or `mixOmics`, as `mlr3extralearners` does for learners, because tuning, bootstrap and permutation analyses refit it many times on high-dimensional blocks and need one start, stopping rule and convergence report under the package's control; wrapping `mixOmics` would also add a Bioconductor dependency. The solver starts deterministically and draws no random numbers, so refits differ only because the data differ. The price is that a converged iteration reaches a start-dependent fixed point of these updates, which need not maximise the reported criterion (by default the mean absolute correlation between block scores); no random restarts are run, and fits that reach the iteration cap are flagged as non-converged. Later components deflate each block by its own score, which keeps a block's successive scores orthogonal, as in RGCCA [@tenenhaus2011rgcca].

**Sequential sparsity tuning.** Instead of searching the block-by-component grid of L1 budgets jointly, the tuners choose one component's budgets at a time, scoring candidates on inner folds, then deflate and continue. This greedy search does not guarantee a joint optimum and its inner scores are optimistic, so nested cross-validation, run directly or as one `batchtools` job per outer fold [@batchtools], estimates held-out association on folds that tuning never sees.

**Bootstrap selection inside the pipeline.** The bootstrap operator refits MB-sPLS at the upstream sparsity on resamples of the training rows, or of whole groups when the task defines them [@efron1994bootstrap], and aligns replicate signs per block. Variables are kept when their percentile interval excludes zero and their mean weight is non-negligible or, under the alternative rule, when their selection frequency reaches a threshold. This follows the spirit of stability selection [@meinshausen2010stability] without its error control; the intervals are stability summaries, not confidence intervals. Stable scores replace the upstream scores at training and prediction, so outer resampling evaluates selection too.

**Deliberately scoped inference.** The operators' built-in permutation tests shuffle rows freely; train-time tests, and fixed-weight tests unless applied to data untouched by fitting, are labelled conditional diagnostics. Confirmatory tests are functions that respect the declared design: a general engine reruns a user-supplied analysis on permuted raw blocks, and a fixed-specification MB-sPLS test, whose component count and sparsity budgets must be set without the tested data, refits in every permutation and returns one global p-value for mutually independent blocks or for a target block independent of the other blocks jointly. No confirmatory per-component test is offered: simple permutation is invalid for later canonical correlations [@winkler2020cca], whose stepwise remedy has, to our knowledge, not been extended to sparse multi-block PLS, and per-component PLS permutation tests showed serious limitations [@danyluik2025pls]. Instead, following the hold-out framework of @monteiro2016holdout, each frozen component's score association can be tested in confirmation data asserted to be independent, with Holm correction across components [@holm1979]; this tests replication of fixed scores, not population rank. With discovery signs the test is one-sided, and replication also requires the sign-oriented correlations to agree with discovery on average. Sampled p-values are never zero [@phipson2010permutation], and Clopper--Pearson intervals for the exceedance probability quantify Monte Carlo error [@clopper1934binomial].

# Research impact statement

`mlr3mbspls` is used in ongoing, not yet published analyses of multimodal psychiatric data at LMU Munich. The package is open source (LGPL-3), with a public development history since August 2025, tagged releases and a changelog [@mlr3mbspls]. For version 0.4.0, the test suite of about 2,300 expectations runs without failures, with a few tests skipped depending on the environment, and continuous integration is configured to run `R CMD check` on Linux, Windows and macOS. An executable vignette, rebuilt by `R CMD check`, walks through a complete analysis. Two simulation scripts in `inst/validation`, run with default settings and fixed seeds, check the type-I error rate of the permutation tests at the 5% level in small reference scenarios. Null rejection rates were 0.045 for the fixed-weight test on data independent of the weights (400 simulations, 199 permutations) and, with 200 simulations and 99 permutations each, 0.055 for the refitted omnibus test with exchangeable rows, 0.070 with whole-unit exchangeability, and 0.045 family-wise for the Holm-corrected confirmation test. All rates met the scripts' pre-specified bound of 0.10, which detects gross rather than small miscalibration, and all 100 strong-signal simulations rejected for both the omnibus and the confirmation test.

# AI usage disclosure

Generative AI coding assistants, Claude Code (Anthropic) and Codex (OpenAI) with several underlying models, were used during the development of the package to help with code review, refactoring, bug fixing, test writing and documentation, and to help draft and revise this paper. The authors directed this work, reviewed and approved all AI-assisted changes, and verified their correctness with the test suite, `R CMD check`, the executable vignette and the simulation-based validation scripts.

# Author contributions

C. S. Vetter developed the MB-sPLS method and its original MATLAB implementation. S. Coors designed and implemented this R package on that basis, including its pipeline integration and inference tools.

# Acknowledgements

This work received no specific funding, and the authors declare no competing interests. We thank the `mlr3` community for discussions and for the infrastructure on which this package builds.

# References
