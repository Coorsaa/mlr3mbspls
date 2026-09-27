# mlr3mbspls: Multi-Block Sparse PLS for mlr3

<div align="center">

[![r-cmd-check](https://github.com/coorsaa/mlr3mbspls/actions/workflows/r-cmd-check.yml/badge.svg)](https://github.com/coorsaa/mlr3mbspls/actions/workflows/r-cmd-check.yml)
[![pkgdown](https://github.com/coorsaa/mlr3mbspls/actions/workflows/pkgdown.yml/badge.svg)](https://github.com/coorsaa/mlr3mbspls/actions/workflows/pkgdown.yml)

</div>

`mlr3mbspls` integrates multi-block sparse partial least squares (MB-sPLS)
with the `mlr3` ecosystem. It provides unsupervised and supervised graph
pipelines, block-aware tasks, nested resampling, descriptive bootstrap
stability, model summaries, visualisations, and a native C++/Armadillo backend.

Development version: **0.4.0**

## Inference boundary

Ordinary bootstrap output describes uncertainty and stability; it is not an
automatic null-hypothesis test. The package provides three deliberately scoped
permutation interfaces:

- `mb_permutation_test()` reruns a complete user-supplied analysis for every
  design-valid shuffle.
- `mbspls_permutation_test()` refits a fixed, pre-specified MB-sPLS analysis and
  returns one global block- or target-association p-value.
- `mb_lc_confirmation_test()` tests pre-specified frozen LC score associations
  in genuinely untouched confirmation observations and applies Holm correction
  across the supplied LC family. It is a directional replication test only
  when the expected correlation signs from discovery are supplied as
  `reference_signs`; otherwise it tests dependence in either direction.

None of these functions turns later components into a generic population-rank
test. Exchangeability units, strata, nuisance-variable handling, and the
scientific null remain study-design responsibilities. Read the
[statistical-validity requirements](inst/STATISTICAL_VALIDITY.md) before reporting
significance.

## Main capabilities

- `TaskMultiBlock()` and packaged synthetic classification, regression, and
  clustering tasks with persistent block metadata.
- `PipeOpMBsPLS`, `PipeOpMBsPLSXY`, and `PipeOpMBsPCA` for sparse multiblock
  representation learning.
- Training-fitted block scaling, site/batch correction, feature suffixing, and
  target-label filtering.
- Sequential component-wise tuning and direct or `batchtools`-backed nested
  cross-validation.
- Group-aware bootstrap stability selection, deterministic L'Ecuyer-CMRG
  streams, sign alignment, and schema-safe frozen scaling helpers.
- Tidy task/model summaries and plots for weights, stability intervals,
  explained variance, scores, correlations, and networks.

## Installation

```r
install.packages(c(
  "mlr3",
  "mlr3pipelines",
  "mlr3cluster",
  "mlr3tuning",
  "data.table",
  "ggplot2"
))
install.packages("remotes")

remotes::install_github("coorsaa/mlr3mbspls")
```

Optional dataset adapters and plots use packages listed in `Suggests`, including
`mixOmics`, `multiblock`, `igraph`, and `ggraph`.

ComBat site correction (`method = "combat"` in `PipeOpSiteCorrection`) needs
the GitHub-only `neuroCombat` package. Install the revision used in continuous
integration with:

```r
install.packages("BiocManager")
BiocManager::install("BiocParallel")
remotes::install_github(
  "Jfortin1/neuroCombat_Rpackage@fbec46a61bc92bedb450b0e44addae4ce6afa934",
  dependencies = NA,
  upgrade = "never"
)
```

pak cannot resolve the Bioconductor remote declared by `neuroCombat`; when
installing with `dependencies = TRUE`, add `"neuroCombat=?ignore"` to the pak
call and install `neuroCombat` as shown above.

## Executable quickstart

This example loads the packaged clustering task, inspects its block structure,
fits a two-component graph, predicts the training rows, and produces two plots.
The complete executable analysis, including nested validation and bootstrap
stability, is in the quickstart vignette.

```r
suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3cluster)
  library(mlr3mbspls)
  library(ggplot2)
})

lgr::lgr$set_threshold("warn")
lgr::get_logger("mlr3")$set_threshold("warn")

task = tsk("mbspls_synthetic_blocks")
blocks = task$block_features()
quality = task$overview()

quality$overview
quality$blocks
lengths(blocks)

site_correction = list(
  block_a = "site_batch",
  block_b = "site_batch",
  block_c = "site_batch"
)
site_methods = list(
  block_a = "partial_corr",
  block_b = "partial_corr",
  block_c = "partial_corr"
)

learner = mbspls_graph_learner(
  learner = lrn("clust.kmeans", centers = 2L),
  task = task,
  site_correction = site_correction,
  site_correction_methods = site_methods,
  ncomp = 2L,
  performance_metric = "mac",
  permutation_test = FALSE,
  bootstrap = FALSE,
  store_train_blocks = TRUE,
  bootstrap_selection = FALSE,
  val_test = "none"
)

learner$train(task)
prediction = learner$predict(task)
report = mbspls_model_summary(learner)

table(prediction$partition)
report$overview
report$components
report$blocks

weight_plot = autoplot(
  learner,
  type = "mbspls_weights",
  source = "weights",
  top_n = 8L
)
variance_plot = autoplot(
  learner,
  type = "mbspls_variance",
  source = "weights",
  show_total = TRUE
)

weight_plot
variance_plot
```

`report$components` lists the training objective and explained variance of
each component. Its `conditional_p_value` column holds the optional train-time
permutation diagnostic (`permutation_test = TRUE`), which conditions on the
fitted preprocessing and hyperparameters; `p_value_scope` states that scope.
Both are `NA` here because the diagnostic was not run. Use the permutation
interfaces below for reportable inference.

## How the pipeline behaves

- **Deterministic solver.** Each component is fitted by block-coordinate
  (Gauss-Seidel) sparse PMD updates that start from the leading direction of
  each block's centred cross-covariance with the other blocks. Fits therefore
  do not depend on the random-number generator: `seed_train` only affects
  permutation draws, and the analysis seed of a permutation test does not
  change an MB-sPLS fit.
- **Convergence.** A component has converged when no block weight vector
  changes by `tol` or more between two sweeps (`PipeOpMBsPLS` and
  `PipeOpMBsPLSXY`: `tol = 1e-4`, at most 600 sweeps). `converged` and
  `iterations` are stored per component, and non-convergence raises a warning.
- **Local optimum.** The criterion is non-convex, so a fit is a local optimum.
  In simulations the deterministic start matched the best of 20 random starts
  in most structured scenarios, but less often for pure noise, very strong
  sparsity, or many more features than rows.
- **Centring.** `PipeOpMBsPLS`, `PipeOpMBsPLSXY`, and `PipeOpMBsPCA` centre
  every retained column with its training mean (`$state$center`) and subtract
  the same means at prediction.
- **Emitted features.** The `LVk_<block>` features always come from the
  training weights, centring, and deflation, at training and at prediction.
  `predict_weights` only selects the weights evaluated in the prediction
  payload (read by the measures and `mbspls_eval_new_data()`) and in
  prediction-side diagnostics. A bootstrap-selection node that is not in
  stability-only mode replaces the LVs by stable-weight LVs, consistently at
  training and at prediction.
- **Block columns.** A declared factor such as `sex` claims its encoded dummy
  columns (for example `sex.m`). Each column can belong to one block only;
  overlapping blocks are an error. The PipeOps, the sequential tuners, site
  correction, and block scaling resolve blocks with the same rule.
- **Site correction.** `PipeOpSiteCorrection` is fitted on the training rows
  and applied unchanged at prediction. Columns it reads at prediction (the
  `"partial_corr"` and `"dir"` columns and the ComBat site) may not be target
  columns; ComBat covariates may use the target, which is read at training
  only. ComBat uses only the site and covariate levels observed in the training
  rows, and the DIR repair is implemented in the package.

## Validation and tuning

The package measures (`mbspls.mac_evwt`, `mbspls.mac`, `mbspls.ev`,
`mbspls.block_ev`) score the prediction payload of each trained model, looked
up by the model's run id. They therefore need stored models: pass
`store_models = TRUE` to `resample()`, `benchmark()`, `tune()`, `ti()`, and
`auto_tuner()` (the same holds for `mbspca.mean_ev`). A component
whose held-out block scores have fewer than two non-degenerate blocks has no
defined association and scores 0 in `mbspls.mac` and `mbspls.mac_evwt`. The
example reuses the quickstart objects and estimates held-out performance of
the fixed configuration.

```r
set.seed(1L)
cv_result = resample(
  task,
  learner,
  rsmp("cv", folds = 3L),
  store_models = TRUE
)
cv_result$aggregate(msrs(c("mbspls.mac_evwt", "mbspls.mac")))
```

`mbspls_nested_cv()` repeats the sequential sparsity tuning of
`TunerSeqMBsPLS` inside every outer analysis split and scores the assessment
rows. Every node with a log environment receives a fresh one per outer fold,
so graphs with bootstrap selection are evaluated as configured and the outer
estimate covers the complete pipeline. The tuners resolve the same block
columns as the final `PipeOpMBsPLS` and centre each inner training fold with
its own means. Inner scores (`inner_scores`, the tuner's `result_y`) were
maximised during tuning and are optimistic.

The tuner's early stopping (`early_stopping = TRUE`; `tuning_early_stop`
defaults to `TRUE` in the nested-CV functions) is a heuristic. It pools
fixed-weight permutation p-values from the inner folds that selected `c`, so
it is selection-optimistic, and `perm_alpha` is a cutoff, not an error rate.

`mbspls_nested_cv_batchtools()` runs one `batchtools` job per outer fold and
returns `list(ids, reg)`:

```r
out = mbspls_nested_cv_batchtools(
  task = task,
  graphlearner = learner,
  rs_outer = rsmp("cv", folds = 5L),
  rs_inner = rsmp("cv", folds = 3L),
  ncomp = 2L,
  tuner_budget = 20L,
  reg_dir = "registry_mbspls_nestedcv"
)
batchtools::submitJobs(reg = out$reg)
batchtools::waitForJobs(reg = out$reg)
nested = collect_mbspls_nested_cv(reg = out$reg)
```

`collect_mbspls_nested_cv()` stops when a requested job failed or has not
finished: the finished folds alone give an incomplete estimate, which is
biased when failures are related to fold difficulty. With
`allow_partial = TRUE` it warns and reports the missing folds as failed rows.

## Bootstrap stability selection

With `bootstrap_selection = TRUE`, `mbspls_graph_learner()` adds a
`PipeOpMBsPLSBootstrapSelect` node. It refits MB-sPLS on bootstrap resamples of
the training data and keeps features whose percentile interval excludes zero
(`selection_method = "ci"`) or whose selection frequency reaches
`frequency_threshold` (`selection_method = "frequency"`). The output describes
stability; it is not a feature-level test.

- Replicate weights are sign-aligned separately for every block, in both
  `align` modes.
- If the task has an mlr3 `group` role, whole groups are resampled by default.
  An explicit `bootstrap_groups` must be equal to or coarser than that role.
  A single resampleable unit is an error, and fewer than 10 units raise a
  warning.
- Intervals are percentile intervals at the stored level `1 - alpha`.
- With the default `stable_weight_source = "training"`, a stable feature must
  be selected by the bootstrap and non-zero in the training fit. Selected
  features with zero training weight are listed in
  `$state$selected_not_in_training`.
- If no component keeps any feature, training stops with guidance; with
  `stability_only = TRUE` it warns instead.
- `seed_bootstrap = NULL` (the default) uses the ambient RNG, so `set.seed()`
  makes the replicates reproducible. An explicit seed gives every replicate its
  own stream, independent of `workers`.

## Proper omnibus permutation inference

For a fixed analysis, `mbspls_permutation_test()` standardises and refits the
MB-sPLS model on every permuted dataset: the `*_sum` statistics refit all
components, the `*_lc1` statistics refit LC1 only. Sampled p-values use
inclusive ties and `(b + 1) / (B + 1)`, so they cannot be zero.

```r
suppressPackageStartupMessages(library(mlr3mbspls))

set.seed(20260831L)
n = 48L
latent = stats::rnorm(n)
raw_numeric_blocks = list(
  clinical = cbind(
    marker = latent + stats::rnorm(n, sd = 0.25),
    noise = stats::rnorm(n)
  ),
  imaging = cbind(
    region = latent + stats::rnorm(n, sd = 0.25),
    noise = stats::rnorm(n)
  )
)

global_test = mbspls_permutation_test(
  blocks = raw_numeric_blocks,
  statistic = "global_lc1",
  n_perm = 99L,
  seed = 20260831L
)

global_test
global_test$p_value
```

```text
Fixed-specification MB-sPLS omnibus Monte Carlo permutation test
Null: Block 'imaging' is independent of the joint fixed block(s) 'clinical' within the supplied exchangeability design.
Statistic = 0.942834; p = 0.01 (0/99 sampled statistics at least as extreme)
Monte Carlo 95% CI for the exceedance probability: [0, 0.03658]
Exchangeability: row; units = 48; strata = 1
Permutation group size: exp(140.7) distinct maps
Seeds: permutation = 20260831; analysis = 1
Statistic: global_lc1; ncomp = 1 (refitted per permutation: 1); standardize = TRUE
Scope: One omnibus p-value for the stated block-independence null. Component statistics after LC1 are descriptive and are not rank-null p-values. Valid only if the statistic, ncomp, c_matrix, standardisation, and solver settings were fixed independently of the tested alignment; otherwise repeat the selection inside mb_permutation_test().
[1] 0.01
```

The result is labelled fixed-specification because its p-value is valid only
if the statistic, `ncomp`, `c_matrix`, standardisation, and solver settings
were chosen without looking at the tested alignment.

As Monte Carlo precision, report the printed interval: the exact
Clopper-Pearson interval for the exceedance probability `b / B`. With no
exceedance in 99 permutations its upper bound is still about 0.037. The Wald
standard error in `monte_carlo_standard_error` is `NA` when `b = 0` or `b = B`.

The MB-sPLS fit is deterministic, so `analysis_seed` does not change this
statistic; report it together with the permutation seed. In an
`mb_permutation_test()` callback with stochastic steps, the analysis seed is
part of the statistic: fix it in advance and never choose among seeds after
seeing results. It selects the L'Ecuyer-CMRG generator, so callbacks that use
forked parallelism remain reproducible.

If imputation, filtering, tuning, sparsity, or component count was selected
from the tested alignment, put the complete procedure inside an
`mb_permutation_test()` callback so every selection step is repeated. Use
`exchangeability_unit`, `within_unit`, and `strata` whenever row-wise shuffling
is not justified, for example `strata = site` to permute within sites when a
shared site effect would otherwise create cross-block dependence. With few
exchangeable units or small strata, the size of the permutation group, not
`n_perm`, limits the attainable p-value; results report that size and warn
when it is small.

## LC-specific confirmation inference

LC-specific p-values require a truly independent confirmation cohort. The
complete score transformation and the LC family must have been frozen before
those observations were examined. Frozen weights fix the sign of every block
score, so supply the expected sign of each pairwise score correlation from
discovery (for example the sign of the discovery-sample correlation) as
`reference_signs`; each LC is then tested one-sided in the discovery direction.
The data below are generated independently of the preceding example, with a
positive expected sign by construction.

```r
suppressPackageStartupMessages(library(mlr3mbspls))

set.seed(20260901L)
n_confirmation = 80L
confirmation_signal = stats::rnorm(n_confirmation)
confirmation_scores = list(
  predictor = cbind(
    LC1 = confirmation_signal + stats::rnorm(n_confirmation, sd = 0.08),
    LC2 = stats::rnorm(n_confirmation)
  ),
  outcome = cbind(
    LC1 = confirmation_signal + stats::rnorm(n_confirmation, sd = 0.08),
    LC2 = stats::rnorm(n_confirmation)
  )
)

confirmed = mb_lc_confirmation_test(
  scores = confirmation_scores,
  independent_confirmation = TRUE,
  permute_blocks = "outcome",
  reference_signs = c("predictor:outcome" = 1),
  n_perm = 99L,
  seed = 20260831L
)

confirmed
```

```text
Independent-confirmation directional permutation tests for frozen MB-sPLS LC score associations
Direction: one-sided in the discovery direction (reference signs)
 component statistic exceedances p_value_raw p_value_holm monte_carlo_conf_low
       LC1  0.994699           0        0.01         0.02            0.0000000
       LC2 -0.121364          86        0.87         0.87            0.7859224
 monte_carlo_conf_high significant_holm direction_agrees replicated
            0.03657574             TRUE             TRUE       TRUE
            0.92818907            FALSE            FALSE      FALSE
Observed signed score correlations:
 component   block_1 block_2 correlation reference_sign oriented_correlation
       LC1 predictor outcome    0.994699              1             0.994699
       LC2 predictor outcome   -0.121364              1            -0.121364
 agrees
   TRUE
  FALSE
Monte Carlo precision: exact 95% Clopper-Pearson interval for the exceedance probability b/B.
Scope: Directional confirmation of fixed learned score associations: one-sided tests of the mean sign-oriented correlation in the discovery direction. An LC replicates only if it is significant and its observed statistic is positive (`replicated`). Not a population-rank test and not valid after reusing confirmation data for fitting or selection.
```

With `reference_signs`, the Holm-adjusted results are directional tests of
fixed score associations. An LC counts as replicated (`replicated`) only if it
is significant and its observed sign-oriented statistic is positive
(`direction_agrees`): with strata or whole-unit designs the permutation null
keeps between-stratum structure fixed, so a significant upper tail can occur
for a pooled correlation of the opposite sign, which shows dependence relative
to the design but not replication. Without them, the statistic is unsigned and
an association in either direction, including one opposite to discovery, can
be significant; the result then establishes dependence, not replication. The
observed signed correlations are always returned in `pairwise_correlations`.
Neither version establishes that the population cross-block rank is at least
two.

## Descriptive bootstrap uncertainty

```r
suppressPackageStartupMessages(library(mlr3mbspls))

set.seed(20260831L)
x = stats::rnorm(80L)
y = 0.6 * x + stats::rnorm(80L, sd = 0.7)
observed = stats::cor(x, y)
replicates = replicate(199L, {
  rows = sample.int(length(x), replace = TRUE)
  stats::cor(x[rows], y[rows])
})

uncertainty = mb_bootstrap_summary(
  replicates = replicates,
  observed = observed,
  conf = 0.95,
  type = "percentile"
)

uncertainty
```

## Supervised MB-sPLS-XY

```r
suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3mbspls)
})

classification_task = tsk("mbspls_synthetic_classif")
classification_learner = mbsplsxy_graph_learner(
  task = classification_task,
  learner = lrn("classif.featureless"),
  ncomp = 1L
)
classification_learner$train(classification_task)
classification_prediction = classification_learner$predict(
  classification_task
)

regression_task = tsk("mbspls_synthetic_regr")
regression_learner = mbsplsxy_graph_learner(
  task = regression_task,
  learner = lrn("regr.featureless"),
  ncomp = 1L
)
regression_learner$train(regression_task)
regression_prediction = regression_learner$predict(regression_task)

classification_prediction
regression_prediction
```

## Documentation and complete workflow

- [Quickstart vignette](vignettes/quickstart.Rmd): every supported inference
  route, nested CV, final models, displayed output, and plots.
- [Statistical-validity requirements](inst/STATISTICAL_VALIDITY.md): leakage,
  exchangeability, nested tuning, metrics, and interpretation.
- [Reproducibility protocol](inst/REPRODUCIBILITY.md): seeds, RNG streams,
  software versions, and reporting requirements.
- [Clinical model card](inst/MODEL_CARD.md): study-specific psychiatry and
  precision-medicine obligations.

To install and open the rendered vignette, also install `knitr` and `rmarkdown`
and build vignettes during installation (a Pandoc installation is required):

```r
install.packages(c("knitr", "rmarkdown"))
remotes::install_github("coorsaa/mlr3mbspls", build_vignettes = TRUE)
vignette("quickstart", package = "mlr3mbspls")
```

## Selected API

| Function | Role |
| --- | --- |
| `TaskMultiBlock()` | Create a task with persistent block membership |
| `mb_task_overview()` | Audit block size, missingness, constants, and target balance |
| `mbspls_graph_learner()` | Build an unsupervised MB-sPLS graph learner |
| `mbsplsxy_graph_learner()` | Build a supervised MB-sPLS-XY graph learner |
| `mbspls_nested_cv()` | Run sequential tuning inside outer validation |
| `mbspls_model_summary()` | Extract tidy fitted-model summaries |
| `mbspls_eval_new_data()` | Evaluate a fitted graph's weights on new data |
| `mb_permutation_test()` | Rerun one complete analysis under valid shuffles |
| `mbspls_permutation_test()` | Run a fixed-specification omnibus MB-sPLS test |
| `mb_lc_confirmation_test()` | Test frozen LCs in independent confirmation data |
| `mb_bootstrap_summary()` | Summarise descriptive bootstrap uncertainty |
| `mbspls_plot_block_weight_ci()` | Plot sign-aligned bootstrap stability intervals |

## Citation

If you use `mlr3mbspls` in academic work, cite:

```text
Coors S, Vetter CS (2026). mlr3mbspls: Multi-Block Sparse Partial Least
Squares for mlr3. R package version 0.4.0.
https://github.com/coorsaa/mlr3mbspls
```

## Getting help and contributing

Ask questions and report bugs in the
[issue tracker](https://github.com/Coorsaa/mlr3mbspls/issues).
[CONTRIBUTING.md](CONTRIBUTING.md) describes what to include in an issue, how
to contribute changes, and how the package is maintained.

All R code, examples, vignettes, and R fences in Markdown follow the pinned
`styler.mlr` guide. Before opening a pull request, run:

```sh
Rscript tools/style.R
Rscript tools/style.R --check
```

## License

LGPL-3
