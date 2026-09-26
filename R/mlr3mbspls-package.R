#' mlr3mbspls: Multi-Block Sparse PLS for mlr3
#'
#' Integration of multi-block sparse partial least squares (MB-sPLS) with the mlr3
#' machine learning framework. This package provides custom PipeOp components for
#' MB-sPLS projection and dimensionality reduction in multi-block data analysis
#' workflows, with support for unsupervised, classification, and regression pipelines.
#'
#' @section Key Features:
#' \itemize{
#'   \item Custom \code{PipeOpMBsPLS} and \code{PipeOpMBsPLSXY} transformers for unsupervised and supervised multi-block representation learning
#'   \item \code{TaskMultiBlock()} factory plus packaged synthetic multi-block tasks with persistent block membership metadata
#'   \item Integration with \pkg{mlr3pipelines} for complex workflows
#'   \item Bootstrap stability selection of sparse weights with group-level resampling
#'   \item Visualization and interpretation tools
#'   \item Block-level task QC via `task$overview()` (with `mb_task_overview()` retained as a wrapper) and tidy reporting via `mbspls_model_summary()`
#'   \item Sequential component-wise tuning and nested resampling
#'   \item Permutation tests with explicit exchangeability designs
#'   \item Statistical-validity, reproducibility, and
#'     clinical model-card checklists installed with the package
#' }
#'
#' @section Fitting:
#' Each MB-sPLS component is fitted by block-coordinate (Gauss-Seidel) sparse
#' PMD updates from a deterministic cross-covariance start, so fits do not
#' depend on the random seed; seeds only govern permutations and bootstrap
#' replicates. The criterion is non-convex and the fit is a local optimum; for
#' pure noise, very strong sparsity, or many more features than rows, other
#' starts can reach a higher objective. Fitted states record per-component
#' convergence. The MB operators centre every block column with its training
#' mean and reuse these means at prediction, and they always compute the
#' emitted LV features from the training fit (in [PipeOpMBsPLS],
#' `predict_weights` only changes the evaluation payload). A downstream
#' [PipeOpMBsPLSBootstrapSelect] that is not in stability-only mode replaces
#' these features with stable LVs, consistently at training and prediction.
#' Block columns declared on original feature names are resolved to their
#' encoded columns by [mb_resolve_block_columns()]; blocks must stay disjoint.
#' The package measures score each resampling iteration on its own prediction
#' and require `store_models = TRUE`.
#'
#' @section Inference boundary:
#' Ordinary bootstrap output is descriptive uncertainty and stability, not a
#' null-hypothesis test. Built-in component-wise permutations are conditional
#' diagnostics and are reported as `conditional_p_value` with a
#' `p_value_scope`. [mb_permutation_test()] reruns a supplied complete analysis
#' and [mbspls_permutation_test()] provides a fixed-specification omnibus
#' MB-sPLS test. [mb_lc_confirmation_test()] tests frozen LC scores only on
#' explicitly independent confirmation observations and applies Holm
#' correction across LCs. It is a directional replication test only when the
#' expected signs from discovery are supplied as `reference_signs`, and then
#' for the mean sign-oriented association across block pairs; otherwise it
#' tests dependence in either direction. None of these tests supports
#' later-LC population-rank claims. All require design-valid exchangeability,
#' and Monte Carlo precision is reported as an exact Clopper-Pearson interval.
#' The package cannot infer the correct grouping, causal estimand, deployment
#' population, or clinical-use requirements from code or data.
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{PipeOpMBsPLS}} / \code{\link{PipeOpMBsPLSXY}}: Main unsupervised and supervised MB-sPLS transformers
#'   \item \code{\link{PipeOpMBsPLSBootstrapSelect}}: Bootstrap stability selection downstream of MB-sPLS
#'   \item \code{\link{PipeOpMBsPCA}}: Multi-block sparse PCA transformer
#'   \item \code{\link{TaskMultiBlock}}: Create multiblock tasks with optional supervision
#'   \item \code{\link{PipeOpSiteCorrection}}: Site/batch correction as a PipeOp
#'   \item \code{\link{mbspls_graph}} / \code{\link{mbsplsxy_graph}}: Construct unsupervised or supervised preprocessing + MB-sPLS graphs
#'   \item \code{\link{mbspls_graph_learner}} / \code{\link{mbsplsxy_graph_learner}}: Wrap graphs as GraphLearners
#'   \item \code{task$overview()} / \code{\link{mb_task_overview}} / \code{\link{mbspls_model_summary}}: Task QC and fitted-model reporting helpers
#'   \item \code{\link{mbspls_eval_new_data}}: Evaluate new data via a trained graph
#'   \item \code{\link{TunerSeqMBsPLS}} / \code{\link{TunerSeqMBsPCA}}: Sequential component-wise sparsity tuning
#'   \item \code{\link{mbspls_nested_cv}} / \code{\link{mbspls_nested_cv_batchtools}}:
#'     Nested resampling utilities
#'   \item \code{\link{mb_permutation_test}} / \code{\link{mbspls_permutation_test}}:
#'     Complete-analysis and fixed-specification MB-sPLS omnibus permutation inference
#'   \item \code{\link{mb_lc_confirmation_test}}: Multiplicity-adjusted frozen-LC
#'     association tests on independent confirmation observations
#' }
#'
#' @name mlr3mbspls-package
#' @aliases mlr3mbspls
"_PACKAGE"
NULL

#' @importFrom Rcpp sourceCpp
#' @useDynLib mlr3mbspls, .registration = TRUE
NULL
