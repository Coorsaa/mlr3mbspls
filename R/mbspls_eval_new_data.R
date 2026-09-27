#' Evaluate MB-sPLS on New Data (EV per block/component, MAC, scores, weights)
#'
#' @description
#' Runs a trained \pkg{mlr3} graph that contains a `PipeOpMBsPLS` on a **new**
#' \code{Task} and returns **evaluation metrics computed on the new data**:
#' block-wise explained variances per component, component-wise totals, the
#' latent-correlation objective (MAC or Frobenius), and the block scores.
#'
#' Internally, this function temporarily attaches a fresh logging environment
#' to the MB-sPLS node (via its `log_env` parameter), seeded with the node's
#' training snapshots, calls \code{gl$predict(task)}, retrieves the payload
#' written by the operator, and augments it with the trained blocks and raw
#' weights from the fitted model.
#'
#' The metrics are computed exactly as in an ordinary prediction of the graph
#' (preprocessing upstream, training centring, deflation order). In particular
#' the evaluated weights follow the node's `predict_weights` setting: with a
#' [PipeOpMBsPLSBootstrapSelect] on the same shared `log_env`, the
#' stability-selected weights the pipeline uses are evaluated. `weights_source`
#' reports which weights were used.
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
#'   \item `ev_comp`  (`numeric [ncomp]`): explained variance per component,
#'     SS-weighted across blocks.
#'   \item `mac_comp` (`numeric [ncomp]`): objective per component on new data
#'     (MAC or Frobenius, matching the trained operator).
#'   \item `T_mat` (`matrix [n_obs x (ncomp*n_blocks)]`): block scores of the
#'     evaluated weights; columns ordered as
#'     \code{LV1_<block1>, ..., LV1_<blockB>, LV2_<block1>, ...}.
#'   \item `blocks` (`character [n_blocks]`): block names (for convenience).
#'   \item `weights_source` (`character(1)`): `"raw"`, `"stable_ci"` or
#'     `"stable_frequency"`, the source of `weights`.
#'   \item `weights` (`list`): the evaluated weights \eqn{w_b^{(k)}};
#'     list-of-lists: component -> block -> named numeric vector.
#'   \item `loadings` (`list`): the evaluated loadings \eqn{p_b^{(k)}} used
#'     for deflation.
#'   \item `weights_raw`, `loadings_raw` (`list`): the raw trained weights and
#'     loadings; identical to `weights`/`loadings` when `weights_source` is
#'     `"raw"`.
#'   \item `blocks_map` (`list`): mapping block -> feature names (training).
#'   \item `ncomp` (`integer(1)`): number of evaluated components.
#'   \item `perf_metric` (`character(1)`): `"mac"` or `"frobenius"` (from training).
#' }
#' Prediction-side diagnostics requested on the node (`val_test`) are included
#' as described in [PipeOpMBsPLS].
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
#' library(mlr3)
#' library(mlr3pipelines)
#' task = tsk("mbspls_synthetic_blocks")
#' blocks = task$block_features()
#' gl = ppl(
#'   "mbspls_graph_learner",
#'   learner = lrn("clust.kmeans", centers = 2L),
#'   blocks = blocks,
#'   ncomp = 1L,
#'   permutation_test = FALSE,
#'   bootstrap = FALSE,
#'   bootstrap_selection = FALSE,
#'   B = 1L,
#'   val_test = "none"
#' )
#' gl$train(task)
#' res = mbspls_eval_new_data(gl, task)
#'
#' # Per-block explained variances on new data:
#' res$ev_block
#'
#' # Per-component totals and MAC:
#' res$ev_comp
#' res$mac_comp
#'
#' # Evaluated weights for LC_01, block "block_a" (see res$weights_source):
#' res$weights[["LC_01"]][["block_a"]]
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

  # A fresh environment keeps the caller's `log_env$last` untouched. It is
  # seeded with the training snapshots (read-only for prediction; lists are
  # copied on modification) so that stability-selected weights published for
  # this run are found exactly as in an ordinary prediction.
  log_env = new.env(parent = emptyenv())
  log_env$warn_overwrite = FALSE
  src_env = if (inherits(old_env_tpl, "environment")) {
    old_env_tpl
  } else if (inherits(old_env_fit, "environment")) {
    old_env_fit
  } else {
    NULL
  }
  if (!is.null(src_env)) {
    for (nm in c("mbspls_states", "mbspls_state", "mbspls_state_last_id")) {
      if (exists(nm, envir = src_env, inherits = FALSE)) {
        assign(nm, get(nm, envir = src_env, inherits = FALSE), envir = log_env)
      }
    }
  }
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

  # --- augment with trained state (blocks, raw weights) ---------------------
  state = .locate_mbspls_model(gl, mbspls_id = node_id)
  payload$weights = payload$weights %||% state$weights
  payload$loadings = payload$loadings %||% state$loadings
  payload$weights_source = payload$weights_source %||% "raw"
  payload$weights_raw = state$weights
  payload$loadings_raw = state$loadings
  payload$blocks_map = state$blocks
  payload$perf_metric = state$performance_metric

  # Tidy names (if not already set in the PipeOp payload)
  K = length(payload$mac_comp %||% payload$weights)
  payload$ncomp = K
  comp_names = paste0("LC_", sprintf("%02d", seq_len(K)))
  if (!is.null(payload$ev_block)) {
    rownames(payload$ev_block) = comp_names
    if (!is.null(payload$blocks)) {
      colnames(payload$ev_block) = payload$blocks
    }
  }
  if (!is.null(payload$ev_comp)) {
    names(payload$ev_comp) = comp_names
  }
  if (!is.null(payload$mac_comp)) {
    names(payload$mac_comp) = comp_names
  }

  payload
}
