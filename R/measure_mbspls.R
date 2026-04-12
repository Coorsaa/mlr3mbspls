# R/measure_mbspls.R
# Compact mlr3 measures for MB-sPLS that read prediction-side diagnostics
# written by PipeOpMBsPLS into `log_env$last`.

#' @title Measures for MB-sPLS (prediction-side)
#' @description
#' These measures score the compact prediction-side payload written by
#' `PipeOpMBsPLS` into `log_env$last` during `$predict()`. They therefore operate
#' on the test split seen by the downstream learner inside resampling / tuning,
#' rather than on training-state diagnostics.
#'
#' The measures are task-type agnostic and can be used with clustering,
#' classification, or regression pipelines as long as the graph contains a
#' `PipeOpMBsPLS` configured with a shared `log_env`.
#'
#' Required payload fields (written by the PipeOp):
#' * `mac_comp`  - numeric vector of per-component latent correlations on test
#'                 data (either MAC or Frobenius, depending on training)
#' * `ev_comp`   - numeric vector of per-component explained variance (test)
#' * `ev_block`  - matrix (K x B) of block-wise EVs on test data
#' * `perf_metric` - `"mac"` or `"frobenius"` (used to normalise Frobenius)
#' * `blocks`    - character vector of block names (used for normalisation)
#'
#' Make sure the same `log_env` is passed into your PipeOp inside the Graph:
#' \preformatted{
#'   metrics_env <- new.env(parent = emptyenv())
#'   po_mb <- po("mbspls", blocks = blocks, ncomp = 3L, log_env = metrics_env)
#'   gl <- as_learner(po_std %>>% po_bs %>>% po_mb %>>% po("learner", lrn("clust.kmeans", centers = 1)))
#' }
#'
#' For `resample()` / `benchmark()`, set `store_models = TRUE` so the trained
#' learner is available when the measure is scored.
#'
#' @family Measures
#' @name mbspls_measures
NULL

# Internal helpers -------------------------------------------------------------

.mbspls_payload = function(learner) {
  if (!inherits(learner, "GraphLearner") || is.null(learner$graph)) {
    return(NULL)
  }
  mbspls_id = tryCatch(
    .mbspls_pipeop_id(learner$graph, where = "learner$graph"),
    error = function(e) NULL
  )
  if (is.null(mbspls_id)) {
    return(NULL)
  }

  po_tpl = learner$graph$pipeops[[mbspls_id]]
  po_fit = tryCatch(learner$model[[mbspls_id]], error = function(e) NULL)

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

.mbspls_measure_key = function(measure) {
  if (is.character(measure) && length(measure) == 1L) {
    id = as.character(measure)
    if (id %in% c("mbspls.mac_evwt", "mbspls.mac", "mbspls.ev", "mbspls.block_ev")) {
      return(id)
    }
    return(NULL)
  }

  if (!inherits(measure, "Measure")) {
    return(NULL)
  }

  if (inherits(measure, "MeasureMBsPLS_EVWeightedMAC")) {
    return("mbspls.mac_evwt")
  }
  if (inherits(measure, "MeasureMBsPLS_MAC")) {
    return("mbspls.mac")
  }
  if (inherits(measure, "MeasureMBsPLS_EV")) {
    return("mbspls.ev")
  }
  if (inherits(measure, "MeasureMBsPLS_BlockEV")) {
    return("mbspls.block_ev")
  }

  id = measure$id %||% NA_character_
  if (id %in% c("mbspls.mac_evwt", "mbspls.mac", "mbspls.ev", "mbspls.block_ev")) {
    return(id)
  }

  NULL
}

.norm_if_frobenius = function(values, payload) {
  perf = if (!is.null(payload$perf_metric)) payload$perf_metric else "mac"
  if (identical(perf, "frobenius")) {
    B = length(payload$blocks %||% character())
    if (is.finite(B) && B >= 2L) {
      values = values / sqrt(choose(B, 2))
    }
  }
  values
}

.mbspls_undefined_measure_condition = function(message, measure, reason) {
  structure(
    list(
      message = message,
      call = NULL,
      measure = measure,
      reason = reason
    ),
    class = c("mbspls_undefined_measure_score", "error", "condition")
  )
}

.mbspls_abort_undefined_measure = function(message, measure, reason) {
  stop(.mbspls_undefined_measure_condition(message, measure, reason))
}

.mbspls_positive_ev_weights = function(ev) {
  ev = as.numeric(ev)
  keep = is.finite(ev)
  if (!any(keep)) {
    return(rep(NA_real_, length(ev)))
  }

  pos = pmax(ev, 0)
  s = sum(pos, na.rm = TRUE)
  if (!is.finite(s) || s <= 1e-12) {
    .mbspls_abort_undefined_measure(
      "EV-weighted MAC is undefined because no component has positive finite prediction-side explained variance. Use 'mbspls.mac' or inspect the fitted model instead of relying on an arbitrary fallback weighting.",
      measure = "mbspls.mac_evwt",
      reason = "nonpositive_ev"
    )
  }

  pos / s
}

#' Compute a scalar MB-sPLS measure directly from a logged payload.
#' @keywords internal
mbspls_measure_score_from_payload = function(payload, measure) {
  key = .mbspls_measure_key(measure)
  if (is.null(key)) {
    stop("Unsupported MB-sPLS measure. Supported measures are: mbspls.mac_evwt, mbspls.mac, mbspls.ev, mbspls.block_ev.", call. = FALSE)
  }
  if (is.null(payload) || !is.list(payload)) {
    return(NA_real_)
  }

  if (identical(key, "mbspls.mac_evwt")) {
    mac = as.numeric(payload$mac_comp)
    ev = as.numeric(payload$ev_comp)
    if (!length(mac) || !length(ev)) {
      return(NA_real_)
    }
    mac = .norm_if_frobenius(mac, payload)
    w = .mbspls_positive_ev_weights(ev)
    if (!length(w) || all(!is.finite(w))) {
      return(NA_real_)
    }
    return(sum(w * mac, na.rm = TRUE))
  }

  if (identical(key, "mbspls.mac")) {
    mac = as.numeric(payload$mac_comp)
    if (!length(mac)) {
      return(NA_real_)
    }
    mac = .norm_if_frobenius(mac, payload)
    return(mean(mac, na.rm = TRUE))
  }

  if (identical(key, "mbspls.ev")) {
    ev = as.numeric(payload$ev_comp)
    if (!length(ev)) {
      return(NA_real_)
    }
    return(mean(ev, na.rm = TRUE))
  }

  evb = payload$ev_block
  if (is.null(evb)) {
    return(NA_real_)
  }
  mean(as.numeric(evb), na.rm = TRUE)
}

mbspls_measure_score_diagnostics = function(payload, measure) {
  key = .mbspls_measure_key(measure)
  if (is.null(key)) {
    stop("Unsupported MB-sPLS measure. Supported measures are: mbspls.mac_evwt, mbspls.mac, mbspls.ev, mbspls.block_ev.", call. = FALSE)
  }

  if (is.null(payload) || !is.list(payload)) {
    return(list(
      score = NA_real_,
      defined = FALSE,
      reason = "missing_payload",
      message = sprintf("Measure '%s' could not be computed because the payload is missing.", key)
    ))
  }

  tryCatch(
    {
      score = as.numeric(mbspls_measure_score_from_payload(payload, key))[1L]
      if (is.finite(score)) {
        return(list(
          score = score,
          defined = TRUE,
          reason = NA_character_,
          message = NA_character_
        ))
      }

      list(
        score = score,
        defined = FALSE,
        reason = "non_finite_score",
        message = sprintf("Measure '%s' returned a non-finite score.", key)
      )
    },
    mbspls_undefined_measure_score = function(e) {
      list(
        score = NA_real_,
        defined = FALSE,
        reason = as.character(e$reason %||% "undefined_measure_score"),
        message = conditionMessage(e)
      )
    }
  )
}

# 1) EV-weighted latent correlation -------------------------------------------

#' @title MB-sPLS EV-weighted latent correlation (prediction-side)
#' @description
#' Aggregates per-component latent correlations (MAC or Frobenius) using the
#' positive part of the prediction-side explained-variance weights from the same
#' split. If all component EVs are non-positive, the measure errors explicitly
#' instead of substituting an arbitrary weighting scheme.
#' @examples
#' \dontrun{
#' msr_evwt = MeasureMBsPLS_EVWeightedMAC$new()
#' }
#' @export
MeasureMBsPLS_EVWeightedMAC = R6::R6Class(
  "MeasureMBsPLS_EVWeightedMAC",
  inherit = mlr3::Measure,
  public = list(
    #' @description Construct the measure.
    #' @param id character(1). Measure ID (default shown).
    initialize = function(id = "mbspls.mac_evwt") {
      super$initialize(
        id           = id,
        minimize     = FALSE,
        range        = c(0, 1),
        task_type    = NA_character_,
        predict_type = NA_character_,
        properties   = c("requires_learner", "requires_no_prediction"),
        packages     = "mlr3"
      )
    }
  ),
  private = list(
    .measure_key = "mbspls.mac_evwt",
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      mbspls_measure_score_from_payload(p, private$.measure_key)
    }
  )
)

# 2) Mean latent correlation across components --------------------------------

#' @title MB-sPLS mean latent correlation across components (prediction-side)
#' @description
#' Mean of per-component latent correlations (MAC or normalised Frobenius).
#' @export
MeasureMBsPLS_MAC = R6::R6Class(
  "MeasureMBsPLS_MAC",
  inherit = mlr3::Measure,
  public = list(
    #' @description Construct the measure.
    #' @param id character(1). Measure ID (default shown).
    initialize = function(id = "mbspls.mac") {
      super$initialize(
        id           = id,
        minimize     = FALSE,
        range        = c(0, 1),
        task_type    = NA_character_,
        predict_type = NA_character_,
        properties   = c("requires_learner", "requires_no_prediction"),
        packages     = "mlr3"
      )
    }
  ),
  private = list(
    .measure_key = "mbspls.mac",
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      mbspls_measure_score_from_payload(p, private$.measure_key)
    }
  )
)

# 3) Mean explained variance across components --------------------------------

#' @title MB-sPLS mean explained variance across components (prediction-side)
#' @description
#' Mean of per-component explained variance on the prediction split.
#' Incremental out-of-sample EV can be negative, so this measure is unbounded
#' below and above in principle.
#' @export
MeasureMBsPLS_EV = R6::R6Class(
  "MeasureMBsPLS_EV",
  inherit = mlr3::Measure,
  public = list(
    #' @description Construct the measure.
    #' @param id character(1). Measure ID (default shown).
    initialize = function(id = "mbspls.ev") {
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
    .measure_key = "mbspls.ev",
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      mbspls_measure_score_from_payload(p, private$.measure_key)
    }
  )
)

# 4) Mean block-wise explained variance across comps & blocks -----------------

#' @title MB-sPLS mean block-wise explained variance (prediction-side)
#' @description
#' Mean of the prediction-side EV matrix across all retained components and all
#' blocks. Incremental out-of-sample EV can be negative, so this measure is
#' unbounded below and above in principle.
#' @export
MeasureMBsPLS_BlockEV = R6::R6Class(
  "MeasureMBsPLS_BlockEV",
  inherit = mlr3::Measure,
  public = list(
    #' @description Construct the measure.
    #' @param id character(1). Measure ID (default shown).
    initialize = function(id = "mbspls.block_ev") {
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
    .measure_key = "mbspls.block_ev",
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      mbspls_measure_score_from_payload(p, private$.measure_key)
    }
  )
)
