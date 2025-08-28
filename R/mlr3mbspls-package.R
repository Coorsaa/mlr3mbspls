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
#'   \item \code{\link{LearnerGraphMB}}: Graph Learner with MB-sPLS capabilities
#'   \item \code{\link{make_task_multiblock}}: Create multi-block tasks
#'   \item \code{\link{get_loadings}}: Extract and format loadings from a model
#'   \item \code{\link{plot_latent_space}}: Visualize the latent space
#' }
#'
#' @name mlr3mbspls-package
#' @aliases mlr3mbspls
"_PACKAGE"
NULL

#' @importFrom Rcpp sourceCpp
#' @useDynLib mlr3mbspls, .registration = TRUE
NULL
