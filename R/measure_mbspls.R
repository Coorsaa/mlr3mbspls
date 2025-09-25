# R/measure_mbspls.R
# Compact set of mlr3 Measures for MB-sPLS that read **test-time** metrics
# from PipeOpMBsPLS via `log_env$last`.

#' @title Measures for MB-sPLS (test-time)
#' @description
#' These measures use the compact payload written by `PipeOpMBsPLS` into
#' `log_env$last` during `$predict()`. They therefore evaluate performance on
#' the **test split** seen by the downstream learner in resampling/tuning.
#'
#' Required payload fields (written by the PipeOp):
#' * `mac_comp`  – numeric vector of per-component latent correlations on test
#'                 data (either MAC or Frobenius, depending on training)
#' * `ev_comp`   – numeric vector of per-component explained variance (test)
#' * `ev_block`  – matrix (K × B) of block-wise EVs on test data
#' * `perf_metric` – `"mac"` or `"frobenius"` (to normalise Frobenius)
#' * `blocks`    – character vector of block names (to count block pairs)
#'
#' Make sure the same `log_env` is passed into your PipeOp inside the Graph:
#' \preformatted{
#'   metrics_env <- new.env(parent = emptyenv())
#'   po_mb <- po("mbspls", blocks = blocks, ncomp = 3L, log_env = metrics_env)
#'   gl    <- as_learner(po_std %>>% po_bs %>>% po_mb %>>% po("learner", lrn("clust.kmeans", centers = 1)))
#' }
#'
#' @family Measures
#' @name mbspls_measures
NULL

# Internal helpers (no export) -------------------------------------------------

.mbspls_payload = function(learner) {
  po = learner$pipeops$mbspls
  env = if (!is.null(po)) po$param_set$values$log_env else NULL
  if (!inherits(env, "environment") || is.null(env$last)) {
    return(NULL)
  }
  env$last
}

.norm_if_frobenius = function(values, payload) {
  perf = if (!is.null(payload$perf_metric)) payload$perf_metric else "mac"
  if (identical(perf, "frobenius")) {
    B = length(payload$blocks %||% character())
    if (is.finite(B) && B >= 2) {
      # normalise Frobenius to [0,1]
      values = values / sqrt(choose(B, 2))
    }
  }
  values
}

`%||%` = function(a, b) if (is.null(a)) b else a

# 1) EV-weighted latent correlation (primary) ----------------------------------

#' @title MB-sPLS EV-weighted latent correlation (test)
#' @description
#' Aggregates per-component test latent correlations (MAC or Frobenius) using
#' explained-variance weights from the same test split:
#' \deqn{\sum_k \left(\frac{EV_k}{\sum_j EV_j}\right) \cdot \text{score}_k.}
#' If Frobenius was used during training, scores are normalised by
#' \eqn{\sqrt{\binom{B}{2}}} to keep the range within \eqn{[0,1]}.
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
        task_type    = "clust",
        predict_type = "partition",
        packages     = "mlr3"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      if (is.null(p)) {
        return(NA_real_)
      }
      mac = as.numeric(p$mac_comp)
      ev = as.numeric(p$ev_comp)
      if (!length(mac) || !length(ev) || all(!is.finite(ev))) {
        return(NA_real_)
      }
      mac = .norm_if_frobenius(mac, p)
      w = ev / (sum(ev, na.rm = TRUE) + 1e-12)
      sum(w * mac, na.rm = TRUE)
    }
  )
)

# 2) Mean latent correlation across components (test) --------------------------

#' @title MB-sPLS mean latent correlation across components (test)
#' @description
#' Mean of per-component test latent correlations (MAC or normalised Frobenius).
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
        task_type    = "clust",
        predict_type = "partition",
        packages     = "mlr3"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      if (is.null(p)) {
        return(NA_real_)
      }
      mac = as.numeric(p$mac_comp)
      if (!length(mac)) {
        return(NA_real_)
      }
      mac = .norm_if_frobenius(mac, p)
      mean(mac, na.rm = TRUE)
    }
  )
)

# 3) Mean explained variance across components (test) --------------------------

#' @title MB-sPLS mean explained variance across components (test)
#' @description
#' Mean of per-component explained variance on the test split.
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
        range        = c(0, 1),
        task_type    = "clust",
        predict_type = "partition",
        packages     = "mlr3"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      if (is.null(p)) {
        return(NA_real_)
      }
      ev = as.numeric(p$ev_comp)
      if (!length(ev)) {
        return(NA_real_)
      }
      mean(ev, na.rm = TRUE)
    }
  )
)

# 4) Mean block-wise explained variance across comps & blocks (test) -----------

#' @title MB-sPLS mean block-wise explained variance (test)
#' @description
#' Mean of the test EV matrix across all components and all blocks.
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
        range        = c(0, 1),
        task_type    = "clust",
        predict_type = "partition",
        packages     = "mlr3"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner, ...) {
      p = .mbspls_payload(learner)
      if (is.null(p)) {
        return(NA_real_)
      }
      evb = p$ev_block
      if (is.null(evb)) {
        return(NA_real_)
      }
      mean(as.numeric(evb), na.rm = TRUE)
    }
  )
)
