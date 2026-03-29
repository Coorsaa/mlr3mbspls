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
