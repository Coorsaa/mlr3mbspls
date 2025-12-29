#' mlr3mbspls: Multi-Block Sparse PLS for mlr3
#'
#' Integration of multi-block sparse partial least squares (MB-sPLS) with the mlr3
#' machine learning framework. This package provides custom PipeOp components for
#' MB-sPLS projection and dimensionality reduction in multi-block data analysis
#' workflows, with support for classification and clustering pipelines.
#'
#' @section Key Features:
#' \itemize{
#'   \item Custom PipeOpMBsPLS for dimensionality reduction
#'   \item Multi-block task creation with block membership tracking
#'   \item Integration with mlr3pipelines for complex workflows
#'   \item Visualization and interpretation tools
#'   \item Hyperparameter tuning support
#' }
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{PipeOpMBsPLS}}: Main PipeOp for MB-sPLS transformation
#'   \item \code{\link{PipeOpSiteCorrection}}: Site/batch correction as a PipeOp
#'   \item \code{\link{mbspls_graph}}: Construct a full preprocessing + MB-sPLS graph
#'   \item \code{\link{mbspls_graph_learner}}: Wrap a graph as a GraphLearner
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
