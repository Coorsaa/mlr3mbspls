# mlr3mbspls 0.4.0 (development)

Changes since 0.3.4. Several fixes change numerical results: fitted weights,
scores, objectives, explained variances, component counts, tuned sparsity
budgets and p-values can differ from 0.3.4 for the same data, settings and
seeds. Refit models and rerun analyses that were produced with earlier
versions.

## Breaking and behaviour changes

### MB-sPLS solver

- The one-component MB-sPLS solver now updates the blocks one at a time, each
  against the latest scores of the other blocks (block-coordinate, or
  Gauss-Seidel, PMD updates). The previous simultaneous updates could cycle
  between two states and could report as converged a solution in which one
  block was anti-aligned with the others. Differences from 0.3.4 are largest
  for sparse data with several latent factors.
- Fits start from a deterministic initialisation, the leading direction of each
  block's centred cross-covariance with the other blocks, and no longer draw
  random numbers. MB-sPLS fits therefore do not depend on `seed_train`,
  `analysis_seed` or the session RNG. `seed_train` now only affects the
  permutations of the training diagnostic. If the fixed power-iteration
  vector misses the cross-covariance (e.g. shared signal in low-variance
  columns next to stronger independent noise), the start is recomputed from
  the column with the largest cross-block covariance before falling back to
  the block's principal axis. The overall sign is fixed so that the largest
  weight of the first block is positive.
- Convergence is declared when no block weight vector changes by `tol` or more
  between two sweeps; previously the solver stopped on the change in the
  reported objective. `PipeOpMBsPLS`, `PipeOpMBsPLSXY`, the refits of
  `PipeOpMBsPLSBootstrapSelect` and `TunerSeqMBsPLS` use at most 600 sweeps and
  `tol = 1e-4`. When the limit is reached, the last iterate is returned. Fitted
  states store `converged` and `iterations` per component, and a warning names
  the components that did not converge.
- The fit is a local optimum of a non-convex criterion. In simulations the
  deterministic start reached the best objective found by 20 random starts in
  most structured scenarios, but less often for pure noise, very strong
  sparsity (budgets close to 1) or many more features than rows. No random
  restarts are run.
- `correlation_method = "spearman"` changes only the reported, tuned and tested
  criterion of each component, not the weight update or the stopping rule.

### Centring and emitted features

- `PipeOpMBsPLS`, `PipeOpMBsPLSXY` and `PipeOpMBsPCA` centre every retained
  block column with its training mean before fitting and subtract the stored
  means (`$state$center`) at prediction. Centring does not change
  first-component weights, but later components, loadings, explained variances
  and scores are now correct for uncentred input, for example after
  `po("blockscale", method = "unit_ssq")`. Graphs with `po("scale")` upstream
  are numerically unaffected. `X_train_blocks` in the training snapshot is
  stored centred.
- `PipeOpMBsPLS` always emits LV features computed from its training weights,
  training means and training deflation, at train and at predict time.
  `predict_weights` now only selects the weights evaluated in the prediction
  payload (`log_env$last`) and by `val_test`. Previously, stable weights could
  silently change the prediction-time features, for example with
  `stability_only = TRUE` and the default `predict_weights = "auto"`. Stable
  LV features reach a learner only through a `PipeOpMBsPLSBootstrapSelect`
  that is not in stability-only mode; it replaces the upstream LV columns
  consistently at train and predict time.
- After a stability-only bootstrap selection, `predict_weights = "auto"`
  evaluates the raw weights, and `"stable_ci"` or `"stable_frequency"` is an
  error in `PipeOpMBsPLS`, `mbspls_graph()` and `mbspls_graph_learner()`.
- MB-sPLS-XY: `emit_y_scores = TRUE` no longer adds target-derived `LVk_.Y`
  columns to the task, which leaked the outcome and made prediction fail. The
  training target scores are stored in `$state$scores_y`. `center_y = FALSE` is
  ignored with a warning; the target block is always centred.
- MB-sPLS-XY no longer appends an auxiliary row-index column to rank-one
  target blocks. The number of supervised components cannot exceed the
  effective rank of the preprocessed target block (one for a binary or
  univariate outcome), and `y_rep` does not increase it.

### Block columns and sparsity budgets

- Block columns are resolved by one rule set, `mb_resolve_block_columns()`, in
  task block views, `task$overview()`, site correction, block scaling, the
  MB-sPLS, MB-sPLS-XY and MB-sPCA operators and the sequential tuners. A
  declared name that is absent from the data expands to its encoded columns
  (e.g. `sex.m` for `sex`), but never claims a column declared in another
  block, and a column that resolves into two blocks is an error. Previously the
  shared helper never matched encoded columns, so factor dummies skipped site
  correction and block scaling but still entered the model, while the
  operators matched prefixes without checking that blocks stay disjoint.
- An unnamed `c_matrix` is matched by position to the declared blocks (plus an
  optional `.target` row for MB-sPLS-XY); use row names to address the
  retained blocks. `NA`, empty or duplicated row names are rejected, also
  for values set through `param_set`. Budgets are validated against the
  structural block width, and budgets above `sqrt(p)` of the retained columns
  are capped and logged instead of failing a resampling fold. Previously an
  unnamed matrix could give a later block's budget to the target in
  MB-sPLS-XY, and duplicated rows silently dropped a budget.

### Measures

- The MB-sPLS and MB-sPCA measures (`mbspls.mac_evwt`, `mbspls.mac`,
  `mbspls.ev`, `mbspls.block_ev`, `mbspca.mean_ev`) now score every resampling
  iteration, benchmarked learner and tuning configuration on its own
  prediction payload, found by the run id of the trained model. Previously all
  of them were scored with the last payload written to the shared `log_env`.
  The measures now have the `"requires_model"` property: `resample()`,
  `benchmark()`, `tune()`, `ti()` and `auto_tuner()` need
  `store_models = TRUE`. A run without a stored payload scores `NA`.
- `mbspls.mac` counts a component whose held-out block scores have fewer than
  two non-degenerate blocks as zero association, as `mbspls.mac_evwt` already
  did in effect.

### Bootstrap stability selection

- `PipeOpMBsPLSBootstrapSelect` resamples whole groups of the task's `group`
  column role when `bootstrap_groups` is `NULL`. An explicit
  `bootstrap_groups` must be equal to or coarser than the group role; remove
  the role to resample finer units.
- Replicate signs are aligned separately for every block in both `align`
  modes; `"score_correlation"` previously applied one sign to all blocks,
  which could produce bimodal weights and intervals spanning zero.
- `seed_bootstrap` defaults to `NULL` in the operator, `mbspls_graph()` and
  `mbspls_graph_learner()` (the operator's `20250921` was only parameter
  metadata and never applied). Replicates then use the session RNG and are
  reproducible with `set.seed()`. When
  `seed_bootstrap` is set, every replicate gets its own L'Ecuyer-CMRG stream,
  so results do not depend on the number of workers.
- An empty stability selection is an error with guidance when
  `stability_only = FALSE`, instead of passing on a task without features.
  With `stability_only = TRUE` it is a warning and no stable weights are
  published.
- Component numbering is preserved: `LVk_<block>` always refers to upstream
  component `k`, and components without stable features emit no LV columns.

### Preprocessing

- `PipeOpSiteCorrection` rejects target columns wherever it reads them at
  prediction time (all `"partial_corr"` and `"dir"` columns and the ComBat
  `site`). ComBat `covariates` may still name the target; they are used at
  training only. `mbspls_preproc_graph()` and `mbsplsxy_graph()` apply the
  same check at construction when a task is supplied.
- A numeric `subgroup` in `PipeOpSiteCorrection` is now interpreted as task row
  ids instead of row positions, so the fitting subgroup is the same in every
  resampling fold. Ids that do not exist in the task are an error. `subgroup`
  also accepts the name of a logical column or
  `list(column = , values = )`.
- The `"dir"` method now implements the geometric repair of Feldman et al.
  (2015) in the package: repair maps are fitted on the training rows and
  applied unchanged at prediction, features are no longer quantised, feature
  names are kept exactly and no extra `protected` column is added.
  `fairmodels` is no longer used.
- With `zero_center = TRUE`, rows of unseen sites are returned on the same
  centred scale as corrected rows. The previously inverted `zero_center`
  documentation is fixed.

### Reporting

- `mbspls_model_summary()` renames `components$p_value` to
  `conditional_p_value` and adds `p_value_scope`, which labels it as the
  conditional train-time diagnostic rather than full-pipeline inference.
  MB-sPLS-XY summaries gain `objective`, `conditional_p_value` and
  `p_value_scope`; MB-sPCA summaries gain `conditional_p_value` and
  `p_value_scope`.
- `aggregate_mbspls_payloads()` defaults to `p_method = "none"` (previously
  `"stouffer"`) and no longer combines fold-wise conditional p-values unless
  asked to. `p_method = "stouffer"` or `"fisher"` is an error unless
  `allow_p_combination = TRUE`; combination then warns about the dependence
  assumptions and is labelled exploratory. (`TunerSeqMBsPLS` still pools fold
  p-values internally, only for its heuristic stopping rule.)
- `mbspls_extract_bootstrap_means()` reads the stored, aligned bootstrap
  summaries. With `filter_method = "ci"`, `filter_level` must equal the stored
  level `1 - alpha`; with `"frequency"`, it defaults to the selector's stored
  `frequency_threshold` (0.6 by default) instead of 0.5.
- `collect_mbspls_nested_cv()` stops when a requested outer-fold job errored,
  expired, was never submitted or has not finished, and reports the affected
  outer splits. `allow_partial = TRUE` warns instead and adds rows with missing
  scores and a status message.

## Confirmatory inference

- New `mb_permutation_test()`: a complete-analysis permutation engine for one
  pre-specified statistic. It permutes the raw selected blocks, reruns a
  user-supplied analysis on the observed data and on every shuffle, and
  supports row, stratum and whole-unit exchangeability. Failed analyses abort
  the test instead of being dropped. `analysis_seed` is reset before every
  analysis call with the L'Ecuyer-CMRG generator, so analyses that use forked
  parallelism stay reproducible; for a stochastic analysis this seed is part of
  the definition of the statistic and must be fixed in advance. The caller's
  RNG state is restored.
- New `mbspls_permutation_test()`: a fixed-specification omnibus MB-sPLS test
  of block independence (`global_lc1`, `global_sum`) or target-block
  independence (`target_lc1`, `target_sum`). Standardisation and the MB-sPLS
  fit are repeated on every permutation: the `*_sum` statistics refit every
  component, the `*_lc1` statistics refit the first component only, and later
  components are summarised descriptively from one fit to the observed data.
  It returns one global p-value, not component-rank p-values. The result is
  labelled as a fixed-specification test and has its own print method showing
  the statistic, component settings and validity scope and reporting
  non-converged observed or permutation fits; a non-converged observed fit
  also raises a warning. Settings selected from the tested data must instead
  be reselected inside `mb_permutation_test()`.
- New `mb_lc_confirmation_test()`: permutation tests of frozen LC scores on
  confirmation observations that were not used for fitting or selection
  (`independent_confirmation = TRUE` must be asserted). It reports
  Holm-adjusted p-values across the supplied LC family, warns when too few
  permutations are drawn for any LC to reach significance after adjustment,
  and returns the observed signed pairwise score correlations
  (`pairwise_correlations`). With `reference_signs` taken from discovery, each
  LC is tested directionally: the statistic is the mean sign-oriented pairwise
  correlation, so a replication shows association in the discovery direction
  on average across block pairs. The replication decision `replicated` also
  requires the observed sign-oriented statistic to be positive
  (`direction_agrees`), because stratified or whole-unit permutations can make
  a pooled correlation of the opposite sign significant relative to the
  design. Without them the unsigned statistic tests dependence in either
  direction, and the result is labelled accordingly.
- Sampled p-values use inclusive ties and `(b + 1) / (B + 1)`. Monte Carlo
  precision is reported as the exact 95% Clopper-Pearson interval for the
  exceedance probability, estimated by `b / B` (`monte_carlo_conf_low`,
  `monte_carlo_conf_high`), and printed. The Wald standard error
  (`monte_carlo_standard_error`) is `NA` when `b` is 0 or `B`, where it would
  be zero.
- Results report the size of the design's permutation group
  (`permutation_group_size`, `log_permutation_group_size`) and warn when
  `n_perm` reaches it, because few exchangeable units or small strata then
  limit the attainable p-value to about `1 / |G|`.
- Blocks may be matrices, data frames, tibbles or data tables. Automatic row
  names are not treated as row ids. Named `exchangeability_unit`,
  `within_unit` and `strata` vectors must match the block row names in order;
  mismatches are errors and vectors are never reordered.

## Statistical validity of diagnostics and summaries

- Prediction-side bootstrap output (`val_test = "bootstrap"`) and
  `mb_bootstrap_summary()` report descriptive uncertainty: the estimate,
  bootstrap mean and bias, standard error, interval, `confidence_level`,
  `interval_type` and requested, effective and failed replicate counts.
  p-value fields are `NA` because an ordinary bootstrap distribution is not a
  null distribution.
- The native permutation diagnostics (training, prediction-side and MB-sPCA)
  always use every requested permutation, inclusive ties and
  `(b + 1) / (B + 1)`. An early-return path previously reported a partial-run
  quantity as a final p-value.
- Train-time permutation p-values are stored with `p_value_scope` in the
  MB-sPLS, MB-sPLS-XY and MB-sPCA states and are documented as conditional
  diagnostics with fixed preprocessing and hyperparameters.
- The MB-sPCA permutation diagnostic tests cross-block association and needs at
  least two blocks. With fewer usable blocks, `PipeOpMBsPCA` and
  `TunerSeqMBsPCA` skip it with a warning and keep the requested number of
  components; `PipeOpMBsPCA` previously truncated extraction to one
  component.
- Prediction-side `val_test` no longer aborts prediction, and with it
  resampling and nested CV, for a component with fewer than two blocks that
  have non-zero weights and non-degenerate scores. Such components get `NA`
  results and the reason in `val_test_status`.
- New exported helpers for corrected permutation p-values
  (`mb_permutation_pvalue()`), group-label permutation
  (`mb_permute_group_labels()`), cluster bootstrap (`mb_cluster_bootstrap()`),
  split checks (`mb_assert_disjoint_groups()`), frozen scaling
  (`mb_fit_scaler()`, `mb_apply_scaler()`), component sign alignment
  (`mb_align_component_signs()`, which flags ambiguous matches by absolute
  cosine similarity), descriptive bootstrap summaries
  (`mb_bootstrap_summary()`) and L'Ecuyer-CMRG streams (`mb_rng_streams()`).
- Prediction-side diagnostics accept `seed_validation`, assign one
  L'Ecuyer-CMRG stream per component and preserve the caller's RNG state.
- The nested-CV entry points and the sequential tuners reject splits that
  share rows or an mlr3 group-role value between analysis and assessment
  partitions.
- Interval-based stability filters require intervals strictly above or below
  zero; intervals touching zero are no longer described as excluding it.
- Seeded training and synthetic data use the Mersenne-Twister, Inversion and
  Rejection generators whatever the session's RNG kind, so results under R's
  default kind are unchanged. Seed 0 is a valid seed; negative, fractional,
  missing or non-numeric seeds are errors.

## Stability selection, plots and evaluation helpers

- The number of exchangeability units is checked before any replicate is
  drawn. Training stops if no unit can vary between replicates (a single unit
  overall or in every stratum), for example when a group role used for
  leave-site-out resampling leaves one site in a training fold. A warning is
  raised, and `few_exchangeability_units` is set, when there are fewer than 10
  units or fewer distinct bootstrap samples than replicates.
- Bootstrap replicates are re-centred on their own resampled rows before
  refitting, and prediction blocks are centred with the upstream training
  means.
- Replicate components are matched to the training components by an exact
  deterministic assignment (`component_matching = "exact_assignment"`), which
  no longer needs the `clue` package. Alignment fallbacks and unresolved signs
  are reported in `alignment_diagnostics`.
- With `stable_weight_source = "training"`, the stable support is the
  intersection of the training support and the bootstrap selection.
  Bootstrap-selected features with zero training weight are logged and
  recorded in `selected_not_in_training`; the per-feature table is stored in
  `selection`.
- New parameter `min_effective_fraction` (default 0.5). Components whose
  accepted replicates fall below this fraction of `B`, or that have none, are
  reported with a warning and listed in `components_below_effective_floor`.
  Non-converged replicate refits are counted and reported.
- The interval summaries in `weights_ci` are percentile intervals of the
  sign-aligned accepted replicates at the stored level `1 - alpha`. The state
  stores `alpha` and the other selection settings.
- Stratified bootstrap draws singleton strata correctly. `stratify_by_block` is
  validated: unknown blocks, blocks dropped upstream and blocks that are not
  dummy-coded are errors; full one-hot and treatment coding are recognised.
- With `bootstrap = FALSE`, the upstream LV columns are kept at prediction time
  as at training time; prediction previously failed with a task mismatch.
- `mbspls_flip_weights()` now flips fitted objects: `PipeOpMBsPLS` and
  `GraphLearner` states, the matching `log_env` entries, bootstrap-selection
  states and the stored prediction payload of the run. An in-place flip is
  refused when a sign-sensitive downstream model was trained on the LV
  columns; `inplace = FALSE` returns flipped copies.
- `mbspls_plot_block_weight_ci()` labels bootstrap intervals with the stored
  level, and `source = "weights"` shows the mean plus or minus one standard
  deviation across fits as a descriptive spread, not a confidence interval.
  `autoplot()` honours `freq_min` for every component and looks up the
  evaluation payload of the learner's own training run.
- `mbspls_eval_new_data()` evaluates the weights that the pipeline uses,
  including stability-selected weights from a shared `log_env`, and returns the
  raw fit in `weights_raw` and `loadings_raw`. The `PipeOpMBsPLS` prediction
  payload gains `weights`, `loadings`, `emitted_weights_source` and, with
  `val_test`, `val_test_status`.

## Site correction and other preprocessing

- `PipeOpSiteCorrection` estimates every correction parameter on the training
  rows and applies it unchanged at prediction, and rejects rank-deficient
  unpenalised `"partial_corr"` designs. ComBat uses only the site and
  covariate levels observed in the training rows, so leave-site-out
  resampling works, and gives clear errors for fewer than two training batches
  and for a `ref_batch` without training rows. The ComBat error messages
  include the install command for the pinned GitHub revision of `neuroCombat`.
- Unpenalised `"partial_corr"` warns when a fitting row is reproduced exactly,
  for example a site with a single fitting row. `unknown_site` is documented
  to apply only to a single categorical site column.
- `PipeOpSiteCorrection` and `PipeOpBlockScaling` update tasks in place, so
  column information, feature types, backend keys and column roles stay
  consistent. `PipeOpBlockScaling` fails on non-finite training predictors and
  on constant predictors for per-feature scaling, and validates its frozen
  state at prediction.
- `PipeOpTargetLabelFilter` filters training rows only. Filtering to a single
  label no longer fails when the task has other factor columns, training level
  sets of factor features are re-applied at prediction, and the
  twoclass/multiclass property is updated after filtering (with mlr3 < 1.7.0
  only when all retained target levels are observed).
- `classif.knngower` and `regr.knngower` share one feature encoder, reject
  missing prediction features under `na_handling = "fail"` and handle missing
  values of single-level ordered features consistently.

## Tuning and nested resampling

- `TunerSeqMBsPLS` and `TunerSeqMBsPCA` resolve and filter block columns
  exactly as the final operators do, on the feature columns the operator
  receives (honouring its `affect_columns`, never the target). Encoded factor
  dummies are now tuned, and the upper bound of each block's `c` uses the
  width the final model fits. Blocks without a usable column are dropped with
  a warning, and the tuned `c_matrix` has one row per retained block.
- Inner folds train the preprocessing graph upstream of the tuned node on the
  fold's training rows only, centre the fold by its own column means, apply
  both to the fold's validation rows, and score explained variance with the
  same denominator as the final model.
- Both tuners check the node's `ncomp` against the rank of each centred
  retained block before searching. Because the solver is deterministic, the
  fits that scored the selected candidate are reused for the fold payloads and
  the deflation instead of being refitted; they equal the fits the operator
  obtains for the same data and `c`. `TunerSeqMBsPLS` no longer runs a
  discarded full-data fit per component. Seeded tuning results can therefore
  differ from earlier versions.
- Both early-stopping rules are documented as heuristics that reuse the data
  that selected `c`, so `perm_alpha` is a cutoff, not an error rate.
  `TunerSeqMBsPLS` evaluates its rule on the inner validation folds and pools
  the dependent fold p-values with a Stouffer combination; the
  `TunerSeqMBsPCA` rule tests cross-block association only. `result_y` is
  optimistic by construction; use nested CV to estimate performance.
- New read-only field `diagnostics` on both tuners, with the resolved blocks,
  per-fold statistics (score, solver convergence and early-stopping p-value)
  and per-component summaries. Both tuners warn when the fold fits of a
  retained component did not converge.
- `TunerSeqMBsPLS` reads `performance_metric` from its parameter set. Errors
  raised while scoring a candidate keep their original message, and MB-sPCA
  candidates with an undefined score rank last. `parallel = "inner"` no longer
  consumes random numbers from the main session, so it proposes the same
  candidates as `parallel = "none"`.
- `mbspls_nested_cv()` and `mbspls_nested_cv_batchtools()` give every node with
  a `log_env` parameter a fresh environment per outer fold, so graphs with
  `PipeOpMBsPLSBootstrapSelect` are evaluated with stability selection as
  configured and the supplied learner is left unchanged. Result rows gain
  `measure_test_n_undefined`, the number of components whose held-out
  association was undefined and scored as zero.
- `mbspls_nested_cv_batchtools()` checks for batchtools before touching the
  file system, defaults `cluster_function` to a one-CPU socket cluster, and
  documents its return value, `list(ids, reg)`. `collect_mbspls_nested_cv()`
  accepts that list as `reg`.

## Reliability and numerical fixes

- The sparse (PMD) update handles tied maxima while preserving the unit L2 norm
  and the L1 budget.
- The MB-sPCA weight update solves the intended constraint (unit L2 norm, L1
  budget in `[1, sqrt(p)]`), which removes runaway weights on heterogeneous,
  unscaled blocks; convergence is evaluated at the updated weights, and
  opposing initial block signs no longer cancel.
- `cpp_mbspls_one_lv()` (internal) returns the objective evaluated at the final
  weights. When the solver converged, it previously returned the objective of
  the preceding iteration, which differed by less than `tol`; no user-facing
  result used this value.
- The multi-component solvers keep score columns only for successful
  components, initialise explained-variance storage and skip an unnecessary
  final deflation. Integer-valued blocks passed to the one-component solver no
  longer read freed memory.
- Stricter validation of `c_matrix`, component ranks, target schemas,
  site-correction specifications and fitted states: invalid mappings,
  non-finite matrices, infeasible budgets, target-rank violations and
  incomplete prediction schemas fail before numerical work. Prediction-side
  native routines reject mismatched row counts, missing weight blocks and
  invalid replicate counts or confidence levels.
- `PipeOpMBsPCA` and `PipeOpMBsPLSXY` implement `.additional_phash_input()`,
  which removes an mlr3pipelines warning during resampling and tuning; all
  three MB operators hash their `blocks`.
- `PipeOpMBsPLS` returns the training `ev_comp` as a named vector, documented
  as SS-weighted across blocks, and `ev_block` as a named matrix; MB-sPCA
  weights and loadings are named by block.
- `task_multiblock_breast_tcga()` now finds the mixOmics blocks, and the
  `mbspls_breast_tcga_classif` and `mbspls_breast_tcga_clust` tasks are
  registered. The optional `multiblock::potato` adapter expands matrix-valued
  blocks correctly and extracts the complete sensory response.
- Missing suggested packages (e.g. for plots or `workers > 1`) raise errors
  with an install hint. Unloading the package also unloads its compiled code.
- Removed unused native routines. Windows builds select the current Armadillo,
  as on other platforms.

## Documentation

- The quickstart vignette is fully executable and produces its output. It runs
  every supported permutation route, descriptive bootstrap uncertainty, nested
  validation, final stability fits and all documented plots without writing
  files to the working directory.
- The README uses compact, executable examples and links to the vignette and
  the installed methodological guidance.
- The installed `STATISTICAL_VALIDITY.md`, `REPRODUCIBILITY.md` and
  `MODEL_CARD.md` describe the statistical-validity requirements, the
  reproducibility protocol and a study checklist. `inst/validation/` contains
  regression scripts for the permutation calculations.
- The help pages of `PipeOpMBsPLS` and `PipeOpMBsPLSXY` document every
  hyperparameter; Rd examples are repaired.

## Packaging and development

- Package version 0.4.0 with a modernised `CITATION`. testthat subprocess
  parallelism is disabled for portable installed-archive checks.
- `fairmodels` was removed from Suggests. `neuroCombat`, needed
  only for ComBat site correction, is available from GitHub only; it stays in
  Suggests without a `Remotes` field, and `R CMD check --as-cran` notes it as a
  suggested package outside the mainstream repositories. Install the revision
  used in CI with
  `remotes::install_github("Jfortin1/neuroCombat_Rpackage@fbec46a61bc92bedb450b0e44addae4ce6afa934")`.
- CI action references and the GitHub-only `neuroCombat` package are pinned to
  commit SHAs, and `neuroCombat` is installed with its hard dependencies only.
  Pull-request pkgdown builds run read-only; only the separate deployment job
  has write permission, and the deployed site keeps `.nojekyll`.
- The pinned `styler.mlr` guide is enforced across package R sources, tests,
  scripts, vignettes and R code in Markdown, in pre-commit and CI.
  `R/RcppExports.R` is committed exactly as `Rcpp::compileAttributes()`
  writes it and is not formatted. `tools/style.R` takes its file list from git,
  so ignored local files are never checked or rewritten, and `--check` lists
  every file that needs formatting.
- Source archives exclude hidden top-level entries, top-level Markdown files
  other than `README.md` and `NEWS.md`, and common local release artefacts; CI
  checks the archive contents. `build.R` formats before generating the
  documentation, installs all dependencies including the pinned `neuroCombat`
  with `--deps`, removes compiled objects and old archives with `--clean`,
  builds vignettes on request, stops when the archive contains files outside
  the package layout, and runs tests against the installed archive in a fresh
  R process.

# mlr3mbspls 0.3.4

## Bug fixes

- `src/mbspca.cpp`: `perm_test_component_mbspca()` no longer uses hardcoded `max_iter=40, tol=1e-4` for permutation refits; these now forward the same solver settings as the main fit (exposed as `max_iter` / `tol` in `PipeOpMBsPCA`).  Added guard for non-positive `c_vec` entries.
- `src/sitecorr.cpp`: replaced diagonal-ratio condition estimate in `cpp_lm_coeff_ridge()` with `arma::rcond(R)` for accurate ill-conditioning detection. Added explicit guard for negative or non-finite `lambda`.
- `R/LearnerClassifKNNGower`, `R/LearnerRegrKNNGower`: emit a warning (rather than silently continuing) when `k` exceeds the training-set size.

## Improvements

- `PipeOpMBsPCA`: `max_iter` (default `60`) and `tol` (default `1e-4`) are now tunable parameters forwarded to both the main solver and the permutation-test refits.
- `PipeOpMBsPLSBootstrapSelect`: `magnitude_threshold` (default `1e-3`) is now an exposed parameter controlling the minimum absolute bootstrap-mean weight required for a feature to pass the CI selection gate (previously hardcoded). A warning is now emitted for each component whose bootstrap replicates are all rejected by the score-correlation gate. `n_eff_by_component` is now stored in `$state` on both success and all-rejected paths.

# mlr3mbspls 0.3.3

## Bug fixes

- `R/measure_mbspls.R`: introduced a typed condition class `mbspls_undefined_measure_score` so that undefined measure scores (e.g. non-positive EV denominator in `mbspls.mac_evwt`) are surfaced as catchable conditions rather than propagating as opaque `NA` or `-Inf` values. Added `mbspls_measure_score_diagnostics()` as an inspectable helper that reports whether a score is defined and, if not, the precise reason.
- `R/TunerSeqMBsPLS.R`: the inner-fold scoring loop now uses `mbspls_measure_score_diagnostics()` in place of direct score extraction; candidates where all folds returned undefined scores are assigned `-Inf` and a structured diagnostic message is attached to the archive, replacing silent score suppression.
- `R/mbspls_nested_cv.R`, `R/mbspls_nested_cv_batch.R`: outer-fold result rows now carry structured score diagnostics (`measure_test_defined`, `measure_test_status`) so downstream `collect_mbspls_nested_cv()` can distinguish truly missing signals from scored-but-undefined results. Introduced shared helper `mbspls_nested_cv_result_row()` to eliminate duplicated logic between the direct and batchtools paths.
- `R/mbspls_model_summary.R`: model summary no longer errors when the log-env is missing bootstrap-selection state; affected fields degrade gracefully to `NA`.

## Improvements

- `mbspls_nested_cv()` and `mbspls_nested_cv_batchtools()`: the outer-evaluation measure is now configurable via a new `measure` argument (accepts an MB-sPLS measure id string or an `mlr3` `Measure` object; default remains `"mbspls.mac_evwt"`). A resolver helper `mbspls_nested_cv_resolve_measure()` validates the supplied measure and binds its payload key.
- `collect_mbspls_nested_cv()`: result table now includes `measure_id`, `measure_key`, `measure_test`, `measure_test_defined`, and `measure_test_status` columns in place of the single implicit `mac_evwt_test` field, making multi-measure comparisons straightforward.
- `R/utils.R`: added `mbspls_metric_summary()` for consistent finite-only mean/SD/range summaries used internally by nested CV reporting.

# mlr3mbspls 0.3.2

- Minor release with stricter validation and guardrails across task handling, pipeops, tuners, evaluation helpers, and native numerical routines.
- Improved multiblock task metadata persistence and overview/reporting ergonomics.
- Improved predict-time consistency checks and explicit erroring instead of silent fallback behavior.

# mlr3mbspls 0.3.1

- Fixed `aggregate_mbspls_payloads()` so component-level MAC, EV, and p-values are aggregated exactly once per fold/component rather than being implicitly duplicated across blocks.
- Fixed `aggregate_mbspls_payloads()` for the one-component monotone-p-value case.
- Fixed `mbspls_nested_cv()` and `mbspls_nested_cv_batchtools()` so the outer evaluation fit uses the tuned `c_matrix` as-is instead of re-running permutation-based early stopping.
- Fixed `mbspls_nested_cv()` to deep-clone the supplied `GraphLearner` per outer fold, preventing mutation of the user-supplied learner object.
- Restored package-build and repository hygiene with `.Rbuildignore`, `.gitignore`, and removal of generated archive/compiled artifacts from the source tree.
- Removed the stale `rlang` `%||%` namespace import; the package consistently uses its internal helper.

## Improvements

- Added `mb_task_overview()` and `task$overview()` for block-wise task QC.
- Added `mbspls_model_summary()` for tidy reporting of fitted MB-sPLS, MB-sPLS-XY, and MB-sPCA models.
- Added early validation for supervised MB-sPLS-XY graph construction so task/learner mismatches fail fast.
- Extended `PipeOpMBsPLSXY` state with Y-side weights/loadings to improve interpretability and downstream reporting.
- Updated README, vignette, package docs, and tests to cover QC/reporting helpers and safer supervised workflows.
