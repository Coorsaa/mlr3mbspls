#' @title Internal utilities for mlr3mbspls
#'
#' @description
#' These are internal utility functions for mlr3mbspls.
#'
#' @keywords internal
#' @name mlr3mbspls_utils
NULL

# Define the null-coalescing operator
`%||%` = function(x, y) if (is.null(x) || length(x) == 0L) y else x


# ------------------------------------------------------------------------------
# Randomness helpers
# ------------------------------------------------------------------------------

#' Execute code with a temporary RNG seed and restore RNG state afterwards.
#'
#' This helper is intentionally implemented without additional dependencies
#' (e.g. withr) and is used to make bootstrap/permutation procedures reproducible
#' without permanently changing the session RNG state.
#'
#' @keywords internal
with_seed_local = function(seed, fn) {
  if (is.null(seed) || length(seed) != 1L || !is.finite(seed)) {
    return(fn())
  }
  seed = as.integer(seed)
  if (!is.finite(seed) || seed <= 0L) {
    return(fn())
  }

  had_seed = exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed = if (had_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL

  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(seed)
  fn()
}

# ------------------------------------------------------------------------------
# Logging helpers (shared env)
# ------------------------------------------------------------------------------

#' Create a unique run id for logging.
#' @keywords internal
make_run_id = function(prefix = "mbspls", log_env = NULL) {
  # timestamp to milliseconds + pid + a monotone counter (per log_env when available)
  ts = format(Sys.time(), "%Y%m%dT%H%M%OS3")
  pid = Sys.getpid()

  ctr = 0L
  if (!is.null(log_env) && inherits(log_env, "environment")) {
    ctr = as.integer(log_env$mbspls_run_counter %||% 0L) + 1L
    log_env$mbspls_run_counter = ctr
  }

  paste(prefix, ts, pid, sprintf("%04d", ctr), sep = "_")
}

#' Store training state in a shared log_env, while keeping a run history.
#'
#' The most recent run is always available in `log_env$mbspls_state` for backward
#' compatibility, and a full history is retained in `log_env$mbspls_states`.
#'
#' @keywords internal
log_env_store_state = function(log_env, payload, warn_overwrite = TRUE) {
  if (is.null(log_env) || !inherits(log_env, "environment")) {
    stop("log_env_store_state: 'log_env' must be an environment.", call. = FALSE)
  }
  if (!is.list(payload)) {
    stop("log_env_store_state: 'payload' must be a list.", call. = FALSE)
  }

  if (is.null(payload$run_id) || !nzchar(as.character(payload$run_id))) {
    payload$run_id = make_run_id("mbspls", log_env)
  }

  if (warn_overwrite && exists("mbspls_state", envir = log_env, inherits = FALSE)) {
    old = log_env$mbspls_state
    if (is.list(old) && !is.null(old$run_id) && !identical(old$run_id, payload$run_id)) {
      warning(
        sprintf("log_env$mbspls_state will be overwritten (old run_id='%s', new run_id='%s').\nFor resampling/parallel runs, prefer a fresh 'log_env' per run; a full history is kept in log_env$mbspls_states.",
          as.character(old$run_id), as.character(payload$run_id)
        ),
        call. = FALSE
      )
    }
  }

  if (is.null(log_env$mbspls_states) || !is.list(log_env$mbspls_states)) {
    log_env$mbspls_states = list()
  }

  log_env$mbspls_states[[as.character(payload$run_id)]] = payload
  log_env$mbspls_state = payload
  log_env$mbspls_state_last_id = as.character(payload$run_id)

  invisible(as.character(payload$run_id))
}

#' Store prediction-side payload in a shared log_env while keeping a run history.
#' @keywords internal
log_env_store_last = function(log_env, payload, run_id = NULL) {
  if (is.null(log_env) || !inherits(log_env, "environment")) {
    stop("log_env_store_last: 'log_env' must be an environment.", call. = FALSE)
  }
  if (!is.list(payload)) {
    stop("log_env_store_last: 'payload' must be a list.", call. = FALSE)
  }

  # Try to infer a run id from the current training snapshot
  if (is.null(run_id) && is.list(log_env$mbspls_state) && !is.null(log_env$mbspls_state$run_id)) {
    run_id = as.character(log_env$mbspls_state$run_id)
  }
  if (!is.null(run_id) && nzchar(run_id)) {
    if (is.null(log_env$mbspls_last) || !is.list(log_env$mbspls_last)) {
      log_env$mbspls_last = list()
    }
    log_env$mbspls_last[[run_id]] = payload
  }

  log_env$last = payload
  invisible(TRUE)
}

# ------------------------------------------------------------------------------
# State validation helpers
# ------------------------------------------------------------------------------

#' Validate the structure of a logged MB-sPLS training snapshot.
#' @keywords internal
assert_mbspls_state = function(st, require_train_blocks = FALSE, where = "log_env$mbspls_state") {
  checkmate::assert_list(st, any.missing = FALSE, min.len = 1L, .var.name = where)

  # required core fields
  checkmate::assert_list(st$blocks, min.len = 1L, names = "strict", .var.name = paste0(where, "$blocks"))
  checkmate::assert_list(st$weights, names = "strict", .var.name = paste0(where, "$weights"))

  # optional but common
  if (!is.null(st$ncomp)) {
    checkmate::assert_integerish(st$ncomp, len = 1L, lower = 0L, .var.name = paste0(where, "$ncomp"))
  }

  # blocks entries should be character vectors
  for (bn in names(st$blocks)) {
    checkmate::assert_character(st$blocks[[bn]], any.missing = FALSE, min.len = 1L,
      .var.name = paste0(where, "$blocks$", bn))
  }

  # optional train blocks
  if (isTRUE(require_train_blocks)) {
    checkmate::assert_list(st$X_train_blocks, min.len = 1L, names = "strict",
      .var.name = paste0(where, "$X_train_blocks"))
  }

  invisible(TRUE)
}

#' Assert that all block features are present in a data.table/data.frame.
#' @keywords internal
assert_blocks_present = function(colnames_dt, blocks_map, context = "task") {
  checkmate::assert_character(colnames_dt, any.missing = FALSE, min.len = 1L, .var.name = "colnames_dt")
  checkmate::assert_list(blocks_map, min.len = 1L, names = "strict", .var.name = "blocks_map")

  missing = lapply(blocks_map, function(cols) setdiff(cols, colnames_dt))
  if (any(lengths(missing) > 0L)) {
    msg = paste0(
      "Missing block features in ", context, ":\n",
      paste0(" - ", names(missing), ": ", vapply(missing, function(x) paste(x, collapse = ", "), character(1)), collapse = "\n")
    )
    stop(msg, call. = FALSE)
  }

  invisible(TRUE)
}
