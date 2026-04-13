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
