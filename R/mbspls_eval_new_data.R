#' Evaluate MB-sPLS on New Data (EV per block/component, MAC, scores, weights)
#'
#' @description
#' Runs a trained \pkg{mlr3} graph that contains a `PipeOpMBsPLS` on a **new**
#' \code{Task} and returns **evaluation metrics computed on the new data**:
#' block-wise explained variances per component, component-wise totals, the
#' latent-correlation objective (MAC or Frobenius), and the block scores.
#'
#' Internally, this function temporarily attaches a logging environment to the
#' MB-sPLS node (via its `log_env` parameter), calls \code{gl$predict(task)},
#' retrieves the payload written by the operator, and augments it with the
#' **trained** weights/loadings/blocks from the fitted model.
#'
#' This ensures perfect parity with the training/prediction implementation
#' (deflation order, preprocessing upstream in the graph, etc.).
#'
#' @param gl (`GraphLearner`)\cr A **trained** graph learner containing a
#'   `PipeOpMBsPLS` node.
#' @param task (`Task`)\cr An \pkg{mlr3} task with **new** data to evaluate on.
#'   The graph's preprocessing will be applied automatically.
#' @param mbspls_id (`character(1)` | `NULL`)\cr Optional id of the MB-sPLS node
#'   in the graph. If `NULL` (default), exactly one node inheriting from
#'   `PipeOpMBsPLS` must be present; otherwise an explicit id is required.
#'
#' @return `list` with the following elements (all computed on the **new** data):
#' \itemize{
#'   \item `ev_block` (`matrix [ncomp x n_blocks]`): explained variance per block and component.
#'   \item `ev_comp`  (`numeric [ncomp]`): total explained variance per component (across blocks).
#'   \item `mac_comp` (`numeric [ncomp]`): objective per component on new data
#'     (MAC or Frobenius, matching the trained operator).
#'   \item `T_mat` (`matrix [n_obs x (ncomp*n_blocks)]`): block scores; columns
#'     ordered as \code{LV1_<block1>, ..., LV1_<blockB>, LV2_<block1>, ...}.
#'   \item `blocks` (`character [n_blocks]`): block names (for convenience).
#'   \item `weights` (`list`): trained weights \eqn{w_b^{(k)}}\; list-of-lists:
#'     component -> block -> named numeric vector (feature weights).
#'   \item `loadings` (`list`): trained loadings \eqn{p_b^{(k)}} used for deflation.
#'   \item `blocks_map` (`list`): mapping block -> feature names (training).
#'   \item `ncomp` (`integer(1)`): number of retained components.
#'   \item `perf_metric` (`character(1)`): `"mac"` or `"frobenius"` (from training).
#' }
#'
#' @details
#' The function **does not** modify the learner permanently. Any existing
#' `log_env` on the MB-sPLS node is restored after the call.
#'
#' @section Errors:
#' An error is thrown if the graph has no `PipeOpMBsPLS` node, if the learner is
#' untrained, or if the MB-sPLS node did not log a payload (e.g., because the
#' id was wrong).
#'
#' @examples
#' \dontrun{
#' # Assume `gl` is a trained GraphLearner with a PipeOpMBsPLS node and
#' # `task_new` is an mlr3::Task with new data:
#' res = mbspls_eval_new_data(gl, task_new)
#'
#' # Per-block explained variances on new data:
#' res$ev_block
#'
#' # Per-component totals and MAC:
#' res$ev_comp
#' res$mac_comp
#'
#' # Access trained weights for LC_02, block "mri":
#' res$weights[["LC_02"]][["mri"]]
#' }
#'
#' @seealso [mlr3pipelines::GraphLearner], [PipeOpMBsPLS]
#' @export
mbspls_eval_new_data = function(gl, task, mbspls_id = NULL) {
  if (!inherits(gl, "GraphLearner")) {
    stop("`gl` must be a trained GraphLearner.", call. = FALSE)
  }
  if (is.null(gl$model)) {
    stop("GraphLearner appears to be untrained (model is NULL).", call. = FALSE)
  }
  if (!inherits(task, "Task")) {
    stop("`task` must be an mlr3 Task.", call. = FALSE)
  }

  # --- locate MB-sPLS node ---------------------------------------------------
  node_id = .mbspls_pipeop_id(gl$graph, mbspls_id = mbspls_id, where = "GraphLearner$graph")

  # --- attach temporary log_env, run predict() -------------------------------
  po_tpl = gl$graph$pipeops[[node_id]]
  po_fit = tryCatch(gl$model[[node_id]], error = function(e) NULL)

  old_env_tpl = tryCatch(po_tpl$param_set$values$log_env, error = function(e) NULL)
  old_env_fit = tryCatch(po_fit$param_set$values$log_env, error = function(e) NULL)
  on.exit({
    if (!is.null(po_tpl)) {
      po_tpl$param_set$values$log_env = old_env_tpl
    }
    if (!is.null(po_fit)) {
      po_fit$param_set$values$log_env = old_env_fit
    }
  }, add = TRUE)

  log_env = new.env(parent = emptyenv())
  po_tpl$param_set$values$log_env = log_env
  if (!is.null(po_fit)) {
    po_fit$param_set$values$log_env = log_env
  }

  invisible(gl$predict(task)) # triggers MB-sPLS to compute EVs/MAC on new data

  payload = log_env$last
  if (is.null(payload)) {
    stop("MB-sPLS node did not log any payload. ",
      "Check the node id and that the PipeOp supports `log_env`.", call. = FALSE)
  }

  # --- augment with trained state (weights/loadings/blocks) ------------------
  state = .locate_mbspls_model(gl, mbspls_id = node_id)
  payload$weights = state$weights
  payload$loadings = state$loadings
  payload$blocks_map = state$blocks
  payload$ncomp = state$ncomp
  payload$perf_metric = state$performance_metric

  # Tidy names (if not already set in the PipeOp payload)
  K = payload$ncomp %||% length(state$weights)
  comp_names = paste0("LC_", sprintf("%02d", seq_len(K)))
  if (!is.null(payload$ev_block)) {
    rownames(payload$ev_block) = comp_names
    if (!is.null(payload$blocks)) {
      colnames(payload$ev_block) = payload$blocks
    }
  }
  if (!is.null(payload$ev_comp)) names(payload$ev_comp) <- comp_names
  if (!is.null(payload$mac_comp)) names(payload$mac_comp) <- comp_names

  payload
}
