# mlr3mbspls 0.4.0 (development)

Changes since 0.3.4 are consolidated for the next release.

## Confirmatory inference

- Added `mb_permutation_test()`, a complete-analysis permutation engine for one
  pre-specified statistic. It permutes raw selected blocks, reruns a supplied
  analysis callback on every shuffle, supports row-, stratum-, and strict
  whole-unit exchangeability, fixes the stochastic analysis seed across
  permutations, preserves caller RNG state, and reports corrected p-values plus
  Monte Carlo precision.
- Added `mbspls_permutation_test()` for refitted omnibus MB-sPLS block- and
  target-association tests. Standardisation and all requested sequential
  components are refitted per permutation. The API returns one global p-value;
  later component statistics are explicitly descriptive rather than presented
  as unsupported rank-null p-values.
- Added `mb_lc_confirmation_test()` for LC-specific permutation tests on frozen
  scores from genuinely independent confirmation observations. It requires an
  explicit independence assertion, supports strict whole-unit and stratum
  exchangeability, and reports Holm-adjusted p-values across the supplied LC
  family. These are replication tests, not population-rank tests.

## Statistical validity

- Prediction-side bootstrap output is now descriptive uncertainty only. It
  reports the estimate, bootstrap mean and bias, standard error, percentile
  interval, confidence level, and requested/effective/failed replicate counts;
  the retained compatibility p-value fields are explicitly `NA`.
- Sampled permutation diagnostics now always use every requested replicate,
  inclusive ties, and `(b + 1) / (B + 1)`. The former early-return path could
  report a partial-run quantity as though it were a final p-value. The same
  full-replicate correction is applied to the MB-sPCA permutation path.
- Cross-fold conditional p-values are no longer combined by default.
  Exploratory Stouffer/Fisher aggregation requires explicit opt-in, warns about
  dependence assumptions, and is labelled as non-confirmatory output.
- Added exported helpers for corrected permutation p-values, group-label
  permutation, cluster bootstrap, group split checks, frozen scaling,
  sign alignment, descriptive bootstrap summaries, and deterministic
  L'Ecuyer-CMRG streams.
- Bootstrap stability selection accepts explicit exchangeability groups and
  assigns a deterministic stream to every replicate, independently of worker
  scheduling. Group and RNG metadata are retained in the fitted state.
- Prediction-side diagnostics accept `seed_validation`, assign one retained
  L'Ecuyer-CMRG stream per component, and preserve the caller's RNG context.
- Nested-CV entry points reject outer splits that leak an `mlr3` group-role
  identifier across analysis and assessment partitions.
- CI-based stability filters now require intervals to be strictly above or
  below zero; intervals touching zero are no longer described as excluding it.

## Reliability and packaging

- Fixed sparse updates at tied maxima, preserving unit L2 norm and the L1
  budget, and prevented opposing initial block signs from cancelling MB-sPCA.
- Bootstrap component matching now uses sequentially deflated scores, and
  seeded parallel resampling preserves the caller's RNG state.
- Sequential tuning preserves the upstream graph, fits feature schemas per
  training fold, and uses the same explained-variance denominator and solver
  settings as the final model. Inner and outer splits reject row/group overlap.
- Site correction applies training-fitted repair maps at prediction, keeps
  unseen-site no-op behavior unchanged, and validates regression rank.
  Target-label filtering applies only to training rows.
- The build script now actually builds requested vignettes and runs requested
  tests against the installed archive in a fresh R process.
- Fixed `cpp_mbspls_one_lv()` so the returned objective is recalculated from
  the final fitted weights. The unfinished refactor had referenced an
  out-of-scope C++ variable and prevented compilation.
- Corrected the MB-sPCA weight update to solve the intended penalised-matrix-
  decomposition constraint: every loading is unit L2 norm with an L1 budget in
  `[1, sqrt(p)]`. This removes the previous runaway-weight failure on
  heterogeneous, unscaled blocks and evaluates convergence at the updated
  weights.
- Corrected the multi-component native solvers so Spearman mode is forwarded to
  every component fit, score columns are retained only for successful
  components, explained-variance storage is initialized, and no unnecessary
  final deflation is attempted.
- Removed the fabricated row-order target direction from MB-sPLS-XY. Requested
  supervised components must now respect the effective target rank; repeating
  target columns is no longer represented as creating rank or an explicit
  target-block multiplier.
- Tightened `c_matrix`, component-rank, target-schema, site-correction, and
  fitted-state validation. Invalid mappings, non-finite matrices, infeasible
  sparsity budgets, target-rank violations, and incomplete prediction schemas
  now fail before numerical work.
- Prediction-side native routines now reject mismatched block row counts,
  missing weight blocks, invalid replicate counts, and invalid confidence
  levels before numerical work.
- `PipeOpBlockScaling` now fails explicitly on non-finite training predictors,
  rejects constant predictors for per-feature scaling, and validates its
  frozen prediction state.
- Fixed the optional `multiblock::potato` adapter so matrix-valued blocks are
  expanded without recursive data-frame columns and the requested sensory
  response is extracted as the complete outcome vector.
- Repaired Rd examples and added regression coverage for numerical, schema,
  adapter, grouping, and inference defects.
- Updated package metadata to 0.4.0, modernized `CITATION`, disabled testthat
  subprocess parallelism for portable installed-tarball checks, and expanded
  the statistical-validity, reproducibility, and model-card documentation.
- CI actions and the GitHub-only `neuroCombat` dependency are pinned to
  immutable revisions. Pull-request pkgdown builds now run read-only; only the
  separate trusted deployment job receives repository write permission.
- Enforced the pinned `styler.mlr` guide across package R sources, tests,
  scripts, vignettes, generated R wrappers, and R examples embedded in
  Markdown. A repository-wide check now runs in pre-commit and CI.
- Rebuilt the quickstart as a fully executable, output-producing vignette. It
  runs every supported permutation route, descriptive bootstrap uncertainty,
  nested validation, final stability fits, and all documented plots without
  writing analysis outputs to the working directory.
- Replaced the duplicated, partly schematic README workflow with compact,
  executable examples and direct links to the complete vignette and installed
  methodological guidance.

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
