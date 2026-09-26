#' Flip signs of MB-sPLS weights/loadings/scores
#'
#' @description
#' MB-sPLS identifies the weights of every component and block only up to
#' sign: the criterion depends on absolute (or squared) cross-block
#' correlations. Flipping a block's weights together with its loadings and
#' scores therefore leaves the fit, the objective and explained variances
#' unchanged, and can be used to orient components for reporting. Per-block
#' sign matrices keep the fit unchanged only for such sign-invariant criteria.
#'
#' All sign-dependent fields that are present are flipped consistently:
#' \code{weights}, \code{loadings}, the stability-selection outputs
#' \code{weights_stable}, \code{loadings_stable}, \code{weights_stable_ci},
#' \code{loadings_stable_ci}, \code{weights_stable_frequency} and
#' \code{loadings_stable_frequency}, the bootstrap summary \code{weights_ci}
#' (means and interval bounds, which are swapped), \code{selection}
#' (training and stable weights), user-supplied \code{weights_boot_draws} and
#' the score columns \code{LV<k>_<block>} of \code{T_mat}, \code{T_mat_train},
#' \code{T_mat_train_stable_all}, \code{T_mat_train_stable_kept} and
#' \code{T_mat_train_kept}. Names and dimensions are preserved. Selection
#' frequencies are sign-free and stay unchanged.
#'
#' \itemize{
#'   \item The \code{list} method flips an MB-sPLS state (as stored by
#'     \code{po("mbspls")}) and returns it.
#'   \item The \code{PipeOpMBsPLS} method flips the trained state and the
#'     matching run entry of its \code{log_env} (read by bootstrap selection and
#'     by \code{predict_weights}), so later predictions are negated
#'     consistently. The stored prediction payload of that run
#'     (\code{log_env$mbspls_last[[run_id]]}, and \code{log_env$last} if it
#'     belongs to the run) has its scores \code{T_mat} flipped as well; its
#'     sign-free statistics are unchanged. The state of a separately trained
#'     \code{PipeOpMBsPLSBootstrapSelect} is not changed; flip a
#'     \code{GraphLearner} to keep both in sync.
#'   \item The \code{GraphLearner} method flips the fitted MB-sPLS state, the
#'     state of every \code{PipeOpMBsPLSBootstrapSelect} node of the same run
#'     and the \code{log_env} run entry with its prediction payload. It works
#'     in place only when every node
#'     downstream of the MB-sPLS node is sign-transparent (a bootstrap-select
#'     node, \code{po("nop")} or a featureless learner): other downstream models
#'     were trained on the original LV columns and would silently change their
#'     predictions, so an error is raised. With \code{inplace = FALSE} the
#'     learner and its \code{log_env} are left unchanged and a flipped copy of
#'     \code{x$model} is returned, e.g. for plotting.
#' }
#'
#' @param x Object containing MB-sPLS state/model
#' @param signs Controls what to flip. One of:
#'   - scalar +1/-1 (applied to ALL components & blocks; -1 means "flip all"),
#'   - numeric vector length K (per component; replicated across blocks),
#'   - numeric matrix K x B with rownames = components (e.g., "LC_01") and
#'     colnames = block names; entries must be +1/-1 (\code{NA} keeps the sign).
#' @param ... Additional arguments passed to methods
#' @return The flipped state for the \code{list} method; for the other methods
#'   the modified object (invisibly) when \code{inplace = TRUE}, otherwise the
#'   flipped state (\code{PipeOpMBsPLS}) or model list (\code{GraphLearner}).
#' @export
mbspls_flip_weights = function(x, signs = -1L, ...) {
  UseMethod("mbspls_flip_weights")
}

#' @rdname mbspls_flip_weights
#' @param flip_boot Also flip stability-selection and bootstrap outputs if
#'   present (default TRUE)
#' @param flip_T Also flip score matrix columns (default TRUE)
#' @export
mbspls_flip_weights.list = function(x, signs = -1L, flip_boot = TRUE, flip_T = TRUE, ...) {
  S = .mbspls_flip_signs(x, signs)
  .mbspls_flip_fields(x, S, flip_boot = flip_boot, flip_T = flip_T)
}

#' @rdname mbspls_flip_weights
#' @param inplace Modify object in-place (TRUE) or return flipped state (FALSE)
#' @export
mbspls_flip_weights.PipeOpMBsPLS = function(x, signs = -1L, inplace = TRUE, flip_boot = TRUE, flip_T = TRUE, ...) {
  if (is.null(x$state) || !length(x$state)) {
    stop("PipeOp has no $state (not trained?)", call. = FALSE)
  }
  S = .mbspls_flip_signs(x$state, signs)
  flipped = .mbspls_flip_fields(x$state, S, flip_boot = flip_boot, flip_T = flip_T)
  if (!isTRUE(inplace)) {
    return(flipped)
  }
  x$state = flipped
  .mbspls_flip_log_env(x$param_set$values$log_env, flipped$run_id, S, flip_boot, flip_T)
  invisible(x)
}

#' @rdname mbspls_flip_weights
#' @param mbspls_id Optional id of the MB-sPLS node when the graph has several.
#' @export
mbspls_flip_weights.GraphLearner = function(x, signs = -1L, inplace = TRUE, flip_boot = TRUE, flip_T = TRUE,
  mbspls_id = NULL, ...) {
  model = x$model
  if (is.null(model)) {
    stop("GraphLearner is not trained; train it before flipping signs.", call. = FALSE)
  }
  id = .mbspls_pipeop_id(x$graph, mbspls_id = mbspls_id, where = "GraphLearner$graph")
  fit_state = model[[id]]
  if (!is.list(fit_state) || !is.list(fit_state$weights)) {
    stop(sprintf("No fitted MB-sPLS state found for node '%s'.", id), call. = FALSE)
  }
  S = .mbspls_flip_signs(fit_state, signs)
  run_id = fit_state$run_id
  model[[id]] = .mbspls_flip_fields(fit_state, S, flip_boot = flip_boot, flip_T = flip_T)

  sel_ids = names(x$graph$pipeops)[vapply(x$graph$pipeops, inherits, logical(1), "PipeOpMBsPLSBootstrapSelect")]
  for (sid in sel_ids) {
    sel_state = model[[sid]]
    if (!is.list(sel_state) || !length(sel_state)) next
    if (!is.null(run_id) && !is.null(sel_state$run_id) && !identical(sel_state$run_id, run_id)) next
    model[[sid]] = .mbspls_flip_fields(sel_state, S, flip_boot = flip_boot, flip_T = flip_T)
  }
  if (!isTRUE(inplace)) {
    return(model)
  }

  blocking = .mbspls_sign_sensitive_downstream(x$graph, id)
  if (length(blocking)) {
    stop(sprintf(
      paste(
        "Cannot flip signs in place: node(s) %s downstream of '%s' were trained on the original LV columns,",
        "so flipping would silently change their predictions. Use inplace = FALSE to obtain flipped states for reporting,",
        "or flip the trained PipeOpMBsPLS before training the downstream learner."
      ),
      paste(sprintf("'%s'", blocking), collapse = ", "), id
    ), call. = FALSE)
  }
  x$model = model
  .mbspls_flip_log_env(x$graph$pipeops[[id]]$param_set$values$log_env, run_id, S, flip_boot, flip_T)
  invisible(x)
}

# Resolve `signs` into a K x B matrix of -1/1/NA with component and block names.
.mbspls_flip_signs = function(x, signs) {
  if (!is.list(x) || !is.list(x$weights) || !is.list(x$loadings)) {
    stop("MB-sPLS state must contain list elements 'weights' and 'loadings'.", call. = FALSE)
  }
  if (length(x$weights) != length(x$loadings)) {
    stop(sprintf(
      "MB-sPLS state is inconsistent: %d weight component(s) but %d loading component(s).",
      length(x$weights),
      length(x$loadings)
    ), call. = FALSE)
  }
  K = as.integer(x$ncomp %||% length(x$weights))
  bn = names(x$blocks) %||% names(x$weights[[1L]])
  B = length(bn)
  comp_names = sprintf("LC_%02d", seq_len(K))
  if (is.null(signs)) {
    signs = -1L
  }
  if (!is.numeric(signs) || any(!is.na(signs) & !signs %in% c(-1, 1))) {
    stop("`signs` entries must be -1 or 1 (NA keeps the sign).", call. = FALSE)
  }

  S = if (length(signs) == 1L && !is.matrix(signs)) {
    matrix(signs, nrow = K, ncol = B, dimnames = list(comp_names, bn))
  } else if (!is.matrix(signs) && length(signs) == K) {
    matrix(signs, nrow = K, ncol = B, dimnames = list(comp_names, bn))
  } else if (is.matrix(signs)) {
    if (!is.null(rownames(signs)) && !is.null(colnames(signs))) {
      missing_rows = setdiff(comp_names, rownames(signs))
      missing_cols = setdiff(bn, colnames(signs))
      if (length(missing_rows) || length(missing_cols)) {
        stop(sprintf(
          "`signs` matrix must name all components (%s) in its rows and all blocks (%s) in its columns.",
          paste(comp_names, collapse = ", "), paste(bn, collapse = ", ")
        ), call. = FALSE)
      }
      signs[comp_names, bn, drop = FALSE]
    } else if (nrow(signs) == K && ncol(signs) == B) {
      matrix(signs, K, B, dimnames = list(comp_names, bn))
    } else if (nrow(signs) == B && ncol(signs) == K) {
      matrix(t(signs), K, B, dimnames = list(comp_names, bn))
    } else {
      stop("`signs` matrix must be K x B or B x K.", call. = FALSE)
    }
  } else {
    stop("Unsupported 'signs' specification.", call. = FALSE)
  }
  storage.mode(S) = "integer"
  S
}

# Flip every sign-dependent field of an MB-sPLS or bootstrap-select state (or
# log_env run entry) according to the sign matrix `S`.
.mbspls_flip_fields = function(x, S, flip_boot = TRUE, flip_T = TRUE) {
  comp_names = rownames(S)
  bn = colnames(S)
  flip_vec = function(v, s) {
    if (!is.numeric(v) || !length(v)) {
      return(v)
    }
    v[] = s * v
    v
  }
  # Component lists are keyed by label when named, otherwise by position.
  flip_nested = function(W) {
    if (!is.list(W) || !length(W)) {
      return(W)
    }
    for (k in seq_along(comp_names)) {
      key = if (is.null(names(W))) k else comp_names[k]
      if ((is.character(key) && !key %in% names(W)) || (is.numeric(key) && key > length(W))) next
      Wk = W[[key]]
      if (!is.list(Wk)) next
      for (b in intersect(bn, names(Wk))) {
        s = S[k, b]
        if (is.na(s) || s == 1L) next
        Wk[[b]] = flip_vec(Wk[[b]], s)
      }
      W[[key]] = Wk
    }
    W
  }

  nested = c("weights", "loadings")
  if (isTRUE(flip_boot)) {
    nested = c(nested, "weights_stable", "loadings_stable", "weights_stable_ci",
      "loadings_stable_ci", "weights_stable_frequency", "loadings_stable_frequency")
  }
  for (field in nested) {
    if (!is.null(x[[field]])) x[[field]] = flip_nested(x[[field]])
  }

  if (isTRUE(flip_T)) {
    for (field in c("T_mat", "T_mat_train", "T_mat_train_stable_all", "T_mat_train_stable_kept", "T_mat_train_kept")) {
      M = x[[field]]
      if (is.null(M) || is.null(colnames(M))) next
      for (k in seq_along(comp_names)) {
        for (b in bn) {
          s = S[k, b]
          col = paste0("LV", k, "_", b)
          if (is.na(s) || s == 1L || !col %in% colnames(M)) next
          M[, col] = s * M[, col]
        }
      }
      x[[field]] = M
    }
  }

  if (isTRUE(flip_boot)) {
    if (!is.null(x$weights_ci)) {
      x$weights_ci = .mbspls_flip_table(x$weights_ci, S,
        value_cols = c("boot_mean", "ci_lower", "ci_upper", "ci_lower_nz", "ci_upper_nz"),
        bounds = list(c("ci_lower", "ci_upper"), c("ci_lower_nz", "ci_upper_nz")))
    }
    for (field in c("selection", "selection_ci", "selection_frequency")) {
      if (!is.null(x[[field]])) {
        x[[field]] = .mbspls_flip_table(x[[field]], S, value_cols = c("training_weight", "stable_weight"))
      }
    }
    if (!is.null(x$weights_boot_draws)) {
      x$weights_boot_draws = .mbspls_flip_table(x$weights_boot_draws, S, value_cols = "weight")
    }
  }
  x
}

# Negate the rows of a component/block-keyed table whose sign is -1; the
# columns in each `bounds` pair are swapped so that lower <= upper.
.mbspls_flip_table = function(tab, S, value_cols, bounds = list()) {
  if (!is.data.frame(tab) || !nrow(tab) || !all(c("component", "block") %in% names(tab))) {
    return(tab)
  }
  is_dt = data.table::is.data.table(tab)
  df = as.data.frame(tab, stringsAsFactors = FALSE)
  comp = as.character(df$component)
  blk = as.character(df$block)
  s = rep(1L, nrow(df))
  for (k in seq_len(nrow(S))) {
    for (b in colnames(S)) {
      if (!is.na(S[k, b])) s[comp == rownames(S)[k] & blk == b] = S[k, b]
    }
  }
  flip = s == -1L
  if (!any(flip)) {
    return(tab)
  }
  for (cc in intersect(value_cols, names(df))) {
    df[[cc]][flip] = -df[[cc]][flip]
  }
  # Negating [lo, hi] gives [-hi, -lo]: swap the (already negated) bounds.
  for (pair in bounds) {
    if (!all(pair %in% names(df))) next
    lo = df[[pair[1L]]][flip]
    df[[pair[1L]]][flip] = df[[pair[2L]]][flip]
    df[[pair[2L]]][flip] = lo
  }
  if (is_dt) data.table::as.data.table(df) else df
}

# Flip the log_env run entry of `run_id` (history and latest snapshot) and its
# stored prediction payload (run-indexed and latest) once.
.mbspls_flip_log_env = function(log_env, run_id, S, flip_boot = TRUE, flip_T = TRUE) {
  if (!inherits(log_env, "environment") || is.null(run_id) || !nzchar(as.character(run_id))) {
    return(invisible(FALSE))
  }
  run_id = as.character(run_id)
  flipped = NULL
  hist = log_env$mbspls_states
  if (is.list(hist) && is.list(hist[[run_id]])) {
    flipped = .mbspls_flip_fields(hist[[run_id]], S, flip_boot = flip_boot, flip_T = flip_T)
    log_env$mbspls_states[[run_id]] = flipped
  }
  latest = log_env$mbspls_state
  if (is.list(latest) && identical(as.character(latest$run_id), run_id)) {
    log_env$mbspls_state = flipped %||% .mbspls_flip_fields(latest, S, flip_boot = flip_boot, flip_T = flip_T)
  }

  # Prediction payloads hold sign-dependent test scores (T_mat).
  flipped_payload = NULL
  payloads = log_env$mbspls_last
  if (is.list(payloads) && is.list(payloads[[run_id]])) {
    flipped_payload = .mbspls_flip_fields(payloads[[run_id]], S, flip_boot = flip_boot, flip_T = flip_T)
    log_env$mbspls_last[[run_id]] = flipped_payload
  }
  last = log_env$last
  if (is.list(last) && identical(as.character(last$run_id), run_id)) {
    log_env$last = flipped_payload %||% .mbspls_flip_fields(last, S, flip_boot = flip_boot, flip_T = flip_T)
  }
  invisible(TRUE)
}

# Nodes downstream of `id` whose trained output would change if the MB-sPLS
# scores were negated. Bootstrap-select nodes are flipped together with the
# fit, and nop or featureless learners ignore the feature values.
.mbspls_sign_sensitive_downstream = function(graph, id) {
  edges = graph$edges
  seen = character(0)
  frontier = id
  while (length(frontier)) {
    nxt = setdiff(unique(edges$dst_id[edges$src_id %in% frontier]), c(seen, id))
    seen = c(seen, nxt)
    frontier = nxt
  }
  transparent = vapply(seen, function(nid) {
    node = graph$pipeops[[nid]]
    if (inherits(node, c("PipeOpMBsPLSBootstrapSelect", "PipeOpNOP"))) {
      return(TRUE)
    }
    inherits(node, "PipeOpLearner") &&
      inherits(node$learner, c("LearnerRegrFeatureless", "LearnerClassifFeatureless", "LearnerClustFeatureless"))
  }, logical(1))
  seen[!transparent]
}
