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

`mlr3mbspls` is an R package that provides MB-sPLS and related methods as steps in machine-learning pipelines of the `mlr3` ecosystem [@mlr3; @mlr3pipelines]. The scores can be combined with data cleaning, correction for differences between measurement sites, and prediction or clustering models, and the whole pipeline can be cross-validated without test data leaking into the fit. The package also measures how stable the selected variables are and tests whether associations could have arisen by chance, respecting study structure such as repeated visits.

# Statement of need

Sparse partial least squares (PLS) and related latent-variable methods are widely used to relate high-dimensional data such as omics assays, brain imaging and clinical measures [@lecao2008spls; @singh2019diablo; @mihalik2022ccapls]. Such analyses face three pitfalls. Their estimates are unstable at typical sample sizes [@helmer2024stability]. Performance estimates can be optimistically biased when any data-dependent step, such as imputation, site correction or variable selection, sees the evaluation data [@ambroise2002selection; @varma2006bias; @kapoor2023leakage]. And permutation tests are valid only when the shuffles respect the structure of the data, such as repeated visits of the same participant or members of the same family [@winkler2015multilevel].

`mlr3mbspls` addresses these pitfalls within `mlr3pipelines`. Every data-dependent step is a pipeline operator that is refitted in each cross-validation fold; bootstrap stability selection shows how consistently variables are selected and can restrict the scores to the stable ones; and confirmatory permutation tests shuffle observations only as the study design allows, for example keeping all visits of a participant together. Each MB-sPLS component gives every block a score built from a small subset of its variables, and these scores become new features for downstream models. The package also offers a supervised variant that treats the outcome as an additional block, multi-block sparse principal component analysis [@witten2009pmd], and block scaling and site correction, including ComBat [@johnson2007combat; @fortin2018harmonization].

The target audience is researchers who analyse multi-block data, for example in psychiatry, neuroimaging or multi-omics, and methodologists comparing such representations with other pipelines.

# State of the field

In R, `mixOmics` provides multi-block sparse PLS with prediction for new samples, and its discriminant form DIABLO adds cross-validated tuning of the number of selected variables [@mixomics; @singh2019diablo]. `RGCCA` implements regularised generalised canonical correlation analysis and its sparse variant SGCCA, including cross-validation, permutation-based tuning of sparsity and bootstrap-based variable selection [@rgcca; @tenenhaus2011rgcca; @tenenhaus2014sgcca]. `PMA` provides sparse multiple canonical correlation analysis (CCA) [@witten2009mcca]; `multiblock` and `ade4` collect many multi-block methods, including sparse multi-block PLS in `multiblock` [@multiblock; @smilde2022multiblock; @bougeard2018ade4]. In Python, `mbpls` offers non-sparse multi-block PLS [@baum2019mbpls], and `cca-zoo` provides sparse multiview CCA usable as a scikit-learn pipeline step [@chapman2021ccazoo; @pedregosa2011sklearn]. Single-block PLS is available as a tidymodels preprocessing step [@recipes] and as `mlr3` learners [@fischer2025mlr3extralearners], but `mlr3pipelines` has no PLS, CCA or multi-block operator.

The MB-sPLS method implemented here was developed by C. S. Vetter, who first implemented it in MATLAB as part of NeuroMiner [@neurominer]; it is closely related to SGCCA and sparse multiple CCA. We did not find R software that (i) provides MB-sPLS scores, bootstrap selection and site correction as pipeline operators that are refitted, along with the sparsity tuning, in every cross-validation fold, or (ii) offers permutation tests for multi-block sparse PLS that follow a declared study design. `RGCCA` and `PMA` tune sparsity with permutations that shuffle all observations freely, and restricted permutations for two-block CCA and PLS exist in a MATLAB toolkit [@mihalik2022ccapls].

We implemented `mlr3mbspls` as a separate `mlr3` extension because we build it on top of `mlr3pipelines` operators. The existing multi-block packages are built around fitting and inspecting a model, and their validation tools resample that model alone; `mlr3mbspls` treats MB-sPLS as one resampled step of a larger pipeline. Adding pipeline operators to those packages would make them depend on `mlr3` infrastructure that does not fit their design goals, whereas `mlr3` distributes such integrations as extension packages [@fischer2025mlr3extralearners].

# Software design

**Operators with training state.** Each operator learns its state, such as centring means, weights and selected variables, from the training data and applies it unchanged to new data. A standalone function would be simpler for interactive use but would leave the separation of training and test data to the user; with operators, `mlr3` resampling, tuning and benchmarking refit every step on each fold's training data. Splits must still keep dependent observations, such as the visits of one participant, together; the package's tuners and nested cross-validation check this when such groups are declared with the data.

**A shared deterministic solver.** Each component is fitted by alternating updates in the style of the penalised matrix decomposition [@witten2009pmd]: each block's sparse weights are updated in turn so that its score covaries maximally with the average score of the other blocks. Later components are fitted to what earlier components leave unexplained, so a block's successive scores are uncorrelated, as in RGCCA [@tenenhaus2011rgcca]. We wrote this solver in C++ [@rcpp; @rcpparmadillo] instead of wrapping `RGCCA` or `mixOmics`, because tuning, bootstrap and permutation analyses refit it many times and should all use the same settings; wrapping `mixOmics` would also add a Bioconductor dependency. The solver starts deterministically from the data and uses no random numbers, so refits differ only because the data differ. The price is that its solution depends on this start and may be a local optimum, not necessarily the global one; no random restarts are run, and fits that do not converge are flagged.

**Sequential sparsity tuning.** Instead of searching the sparsity levels of all components jointly, the tuners choose them one component at a time on inner cross-validation folds, then remove that component from the data and continue. This greedy search covers a far smaller space but does not guarantee a joint optimum, and its inner scores are optimistic; nested cross-validation therefore estimates the association on held-out data that tuning never sees.

**Bootstrap selection inside the pipeline.** The bootstrap operator refits MB-sPLS with the same sparsity on resamples of the training observations, or of whole groups when these are declared [@efron1994bootstrap]. Depending on the chosen rule, it keeps variables whose bootstrap interval for the weight stays clearly away from zero, or those selected in a large share of resamples. This follows the spirit of stability selection [@meinshausen2010stability] without its error control, and these intervals summarise stability; they are not confidence intervals. Because the stable scores replace the original ones at training and prediction, outer resampling evaluates the selection step too.

**Deliberately scoped inference.** The permutation tests built into the pipeline steps shuffle observations freely and are labelled as diagnostics unless applied to data not used for fitting. Confirmatory tests are separate functions that shuffle only as the declared study design allows. One reruns any user-supplied analysis on permuted data. The other refits MB-sPLS in every permutation, with the number of components and the sparsity chosen without the tested data, and returns a single p-value for independence between blocks. We offer no permutation test per component: simple permutation is invalid for canonical correlations after the first [@winkler2020cca], the correction proposed there has not, to our knowledge, been adapted to sparse multi-block PLS, and per-component PLS tests have shown serious limitations [@danyluik2025pls]. Instead, following @monteiro2016holdout, each component can be tested for replication in independent confirmation data with its weights held fixed, using Holm correction across components [@holm1979].

# Research impact statement

The MB-sPLS method has been applied to multimodal data from the PRONIA study, linking aggression potential in early psychosis to social adversity, brain structure and polygenic risk [@weyer2026aggression], and `mlr3mbspls` is used in ongoing, not yet published analyses of multimodal psychiatric data at LMU Munich. The package is open source (LGPL-3) and has been developed publicly since August 2025, with tagged releases and a changelog [@mlr3mbspls]. Its automated tests run in continuous integration on the major operating systems, an executable vignette walks through a complete analysis, and reproducible simulation scripts check in small scenarios that the permutation tests show no gross excess of false positives when there is no effect and do detect strong effects.

# AI usage disclosure

Generative AI coding assistants, Claude Code (Anthropic) and Codex (OpenAI) with several underlying models, were used during the development of the package to help with code review, refactoring, bug fixing, test writing and documentation, and to help draft and revise this paper. The authors directed this work, reviewed and approved all AI-assisted changes, and verified their correctness with the test suite, `R CMD check`, the executable vignette and the simulation-based validation scripts.

# Author contributions

C. S. Vetter developed the MB-sPLS method and its original MATLAB implementation in NeuroMiner. S. Coors designed and implemented this R package on that basis, including its pipeline integration and inference tools.

# Acknowledgements

This work received no specific funding, and the authors declare no competing interests. We thank the `mlr3` community for discussions and for the infrastructure on which this package builds.

# References
