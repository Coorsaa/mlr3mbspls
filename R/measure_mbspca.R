#' @title MB-SPCA Mean Explained Variance
#' @description
#' A custom [mlr3::Measure] that computes the mean explained variance of the latent variables in MB-SPCA.
#'
#' @section Construction:
#' ```
#' MeasureMBSPCAMEV$new()
#' ```
#'
#' @section Methods:
#' - `$score(prediction, task, learner, ...)`: Computes the measure based on the prediction and task.
#'
#' @examples
#' \dontrun{
#' measure = MeasureMBSPCAMEV$new()
#' }
#'
#' @export
MeasureMBSPCAMEV = R6::R6Class(
  "MeasureMBSPCAMEV",
  inherit = mlr3::Measure,
  public = list(
    #' @description Construct the measure.
    #' @param id character(1). Measure ID (default shown).
    initialize = function() {
      super$initialize(
        id           = "mbspca.mean_ev",
        minimize     = FALSE,
        range        = c(0, 1),
        task_type    = "clust",
        predict_type = "partition"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner, ...) {
      po  = learner$pipeops$mbspca
      st  = po$state
      if (is.null(st$explained_variance)) return(0)
      mean(st$explained_variance, na.rm = TRUE)
    }
  )
)
