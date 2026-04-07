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
#'   \item Visualization and interpretation tools
#'   \item Block-level task QC via `task$overview()` (with `mb_task_overview()` retained as a wrapper) and tidy reporting via `mbspls_model_summary()`
#'   \item Hyperparameter tuning support
#' }
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{PipeOpMBsPLS}} / \code{\link{PipeOpMBsPLSXY}}: Main unsupervised and supervised MB-sPLS transformers
#'   \item \code{\link{TaskMultiBlock}}: Create multiblock tasks with optional supervision
#'   \item \code{\link{PipeOpSiteCorrection}}: Site/batch correction as a PipeOp
#'   \item \code{\link{mbspls_graph}} / \code{\link{mbsplsxy_graph}}: Construct unsupervised or supervised preprocessing + MB-sPLS graphs
#'   \item \code{\link{mbspls_graph_learner}} / \code{\link{mbsplsxy_graph_learner}}: Wrap graphs as GraphLearners
#'   \item \code{task$overview()} / \code{\link{mb_task_overview}} / \code{\link{mbspls_model_summary}}: Task QC and fitted-model reporting helpers
#'   \item \code{\link{mbspls_eval_new_data}}: Evaluate new data via a trained graph
#'   \item \code{\link{mbspls_nested_cv}} / \code{\link{mbspls_nested_cv_batchtools}}:
#'     Nested resampling utilities
#' }
#'
#' @name mlr3mbspls-package
#' @aliases mlr3mbspls
"_PACKAGE"
NULL

#' @importFrom Rcpp sourceCpp
#' @useDynLib mlr3mbspls, .registration = TRUE
NULL
