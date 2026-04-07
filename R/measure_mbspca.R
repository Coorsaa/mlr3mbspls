#' @keywords internal
#' @noRd
.mbspca_payload = function(learner) {
  if (!inherits(learner, "GraphLearner") || is.null(learner$graph)) {
    return(NULL)
  }
  mbspca_id = tryCatch(
    .find_pipeop_id_by_class(learner$graph, class_name = "PipeOpMBsPCA", where = "learner$graph"),
    error = function(e) NULL
  )
  if (is.null(mbspca_id)) {
    return(NULL)
  }

  po_tpl = learner$graph$pipeops[[mbspca_id]]
  po_fit = tryCatch(learner$model[[mbspca_id]], error = function(e) NULL)

  envs = Filter(
    function(x) inherits(x, "environment"),
    list(
      tryCatch(po_fit$param_set$values$log_env, error = function(e) NULL),
      tryCatch(po_tpl$param_set$values$log_env, error = function(e) NULL)
    )
  )
  if (!length(envs)) {
    return(NULL)
  }

  run_ids = unique(Filter(
    function(x) !is.null(x) && nzchar(as.character(x)),
    list(
      tryCatch(po_fit$state$run_id %||% NULL, error = function(e) NULL),
      tryCatch(po_tpl$state$run_id %||% NULL, error = function(e) NULL)
    )
  ))

  for (env in envs) {
    for (run_id in run_ids) {
      by_id = env$mbspls_last[[as.character(run_id)]] %||% NULL
      if (is.list(by_id)) {
        return(by_id)
      }
    }
    if (is.list(env$last)) {
      return(env$last)
    }
  }

  NULL
}

.mbspca_measure_key = function(measure) {
  if (is.character(measure) && length(measure) == 1L) {
    id = as.character(measure)
    if (identical(id, "mbspca.mean_ev")) {
      return(id)
    }
    return(NULL)
  }

  if (!inherits(measure, "Measure")) {
    return(NULL)
  }

  if (inherits(measure, "MeasureMBSPCAMEV")) {
    return("mbspca.mean_ev")
  }
  id = measure$id %||% NA_character_
  if (identical(id, "mbspca.mean_ev")) {
    return(id)
  }
  NULL
}

#' Compute the scalar MB-sPCA EV measure from a logged payload.
#' @keywords internal
mbspca_measure_score_from_payload = function(payload, measure = "mbspca.mean_ev") {
  key = .mbspca_measure_key(measure)
  if (!identical(key, "mbspca.mean_ev")) {
    stop("Unsupported MB-sPCA measure. Supported measure is: mbspca.mean_ev.", call. = FALSE)
  }
  if (is.null(payload) || !is.list(payload)) {
    return(NA_real_)
  }
  ev = as.numeric(payload$ev_comp)
  if (!length(ev)) {
    return(NA_real_)
  }
  mean(ev, na.rm = TRUE)
}

#' @title MB-sPCA Mean Explained Variance
#' @description
#' A custom [mlr3::Measure] that computes the mean prediction-side explained
#' variance of the retained MB-sPCA components from the payload written by
#' `PipeOpMBsPCA` into `log_env$last` during `$predict()`.
#'
#' The measure is task-type agnostic and can therefore be used with clustering,
#' classification, or regression pipelines that contain a `PipeOpMBsPCA` node.
#' For `resample()` / `benchmark()`, set `store_models = TRUE` so the trained
#' learner is available during scoring.
#'
#' @section Construction:
#' ```
#' MeasureMBSPCAMEV$new()
#' ```
#'
#' @section Methods:
#' - `$score(prediction, task, learner, ...)`: Computes the measure from the
#'   MB-sPCA prediction payload.
#'
#' @examples
#' \dontrun{
#' measure = MeasureMBSPCAMEV$new()
#' }
#' @export
MeasureMBSPCAMEV = R6::R6Class(
  "MeasureMBSPCAMEV",
  inherit = mlr3::Measure,
  public = list(
    #' @description Construct the measure.
    #' @param id character(1). Measure ID.
    initialize = function(id = "mbspca.mean_ev") {
      super$initialize(
        id           = id,
        minimize     = FALSE,
        range        = c(-Inf, Inf),
        task_type    = NA_character_,
        predict_type = NA_character_,
        properties   = c("requires_learner", "requires_no_prediction"),
        packages     = "mlr3"
      )
    }
  ),
  private = list(
    .measure_key = "mbspca.mean_ev",
    .score = function(prediction, task, learner, ...) {
      p = .mbspca_payload(learner)
      mbspca_measure_score_from_payload(p, private$.measure_key)
    }
  )
)
