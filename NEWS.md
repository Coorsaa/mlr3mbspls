# mlr3mbspls 0.3.4

## Bug fixes

- `src/mbspca.cpp`: `perm_test_component_mbspca()` no longer uses hardcoded `max_iter=40, tol=1e-4` for permutation refits; these now forward the same solver settings as the main fit (exposed as `max_iter` / `tol` in `PipeOpMBsPCA`).  Added guard for non-positive `c_vec` entries.
- `src/sitecorr.cpp`: replaced diagonal-ratio condition estimate in `cpp_lm_coeff_ridge()` with `arma::rcond(R)` for accurate ill-conditioning detection. Added explicit guard for negative or non-finite `lambda`.
- `R/LearnerClassifKNNGower`, `R/LearnerRegrKNNGower`: emit a warning (rather than silently continuing) when `k` exceeds the training-set size.

## Improvements

- `PipeOpMBsPCA`: `max_iter` (default `60`) and `tol` (default `1e-4`) are now tunable parameters forwarded to both the main solver and the permutation-test refits.
- `PipeOpMBsPLSBootstrapSelect`: `magnitude_threshold` (default `1e-3`) is now an exposed parameter controlling the minimum absolute bootstrap-mean weight required for a feature to pass the CI selection gate (previously hardcoded). A warning is now emitted for each component whose bootstrap replicates are all rejected by the score-correlation gate.

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
