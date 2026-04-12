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




#' Format a character vector for concise error messages.
#' @keywords internal
mb_format_truncated = function(x, max_items = 20L) {
  x = as.character(x %||% character(0))
  if (!length(x)) {
    return("")
  }
  max_items = as.integer(max_items %||% 20L)
  max_items = if (is.finite(max_items) && max_items >= 1L) max_items else 20L
  shown = utils::head(x, max_items)
  out = paste(shown, collapse = ", ")
  if (length(x) > max_items) {
    out = sprintf("%s, ... (+%d more)", out, length(x) - max_items)
  }
  out
}

#' Assert that a dataset contains all columns required by a trained model.
#' @keywords internal
mb_assert_columns_present = function(colnames_dt, required, context = "data", hint = NULL) {
  checkmate::assert_character(colnames_dt, any.missing = FALSE, .var.name = "colnames_dt")
  checkmate::assert_character(required, any.missing = FALSE, .var.name = "required")

  required = unique(required)
  missing = setdiff(required, colnames_dt)
  if (length(missing)) {
    msg = sprintf(
      "%s is missing %d trained feature(s): %s",
      context,
      length(missing),
      mb_format_truncated(missing)
    )
    if (!is.null(hint) && length(hint) == 1L && nzchar(as.character(hint))) {
      msg = paste0(msg, "\nFix: ", as.character(hint))
    }
    stop(msg, call. = FALSE)
  }
  invisible(TRUE)
}

#' Align a numeric vector to trained feature names.
#' @keywords internal
mb_align_named_numeric = function(v, cols, context = "vector", allow_null = FALSE, zero_if_null = FALSE) {
  checkmate::assert_character(cols, any.missing = FALSE, .var.name = "cols")

  if (is.null(v)) {
    if (isTRUE(zero_if_null)) {
      return(stats::setNames(numeric(length(cols)), cols))
    }
    if (isTRUE(allow_null)) {
      return(NULL)
    }
    stop(sprintf("%s is NULL; the trained model does not contain coefficients for these features.", context), call. = FALSE)
  }

  if (!is.numeric(v)) {
    stop(sprintf("%s must be numeric.", context), call. = FALSE)
  }

  if (!is.null(names(v))) {
    missing = setdiff(cols, names(v))
    if (length(missing)) {
      stop(sprintf(
        "%s is missing %d trained feature name(s): %s",
        context,
        length(missing),
        mb_format_truncated(missing)
      ), call. = FALSE)
    }
    out = as.numeric(v[cols])
  } else {
    out = as.numeric(v)
    if (length(out) != length(cols)) {
      stop(sprintf(
        "%s has length %d but %d trained feature(s) are required.",
        context,
        length(out),
        length(cols)
      ), call. = FALSE)
    }
  }

  if (anyNA(out) || any(!is.finite(out))) {
    stop(sprintf("%s contains NA/Inf values after alignment.", context), call. = FALSE)
  }

  stats::setNames(out, cols)
}

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

  env_warn_overwrite = log_env$warn_overwrite
  if (is.null(env_warn_overwrite)) {
    env_warn_overwrite = TRUE
  }
  warn_overwrite = isTRUE(warn_overwrite) && isTRUE(env_warn_overwrite)

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

  # Only persist run-indexed prediction payloads when run_id is explicit.
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

#' Resolve an MB-sPLS state from a shared log_env, preferring a specific run_id.
#' @keywords internal
.mbspls_state_from_env = function(log_env, run_id = NULL, require_train_blocks = FALSE, where = "log_env") {
  if (is.null(log_env) || !inherits(log_env, "environment")) {
    stop(sprintf("%s must be an environment.", where), call. = FALSE)
  }

  requested = if (is.null(run_id) || !nzchar(as.character(run_id))) NULL else as.character(run_id)
  hist = log_env$mbspls_states %||% NULL
  st = NULL

  if (!is.null(requested) && is.list(hist) && length(hist)) {
    st = hist[[requested]] %||% NULL

    if (is.null(st)) {
      latest = log_env$mbspls_state %||% NULL
      latest_id = if (is.list(latest)) latest$run_id %||% NULL else NULL
      if (!is.null(latest) && identical(as.character(latest_id), requested)) {
        st = latest
      } else {
        stop(sprintf("mbspls_state for run_id='%s' not found in %s$mbspls_states.", requested, where), call. = FALSE)
      }
    }
  }

  if (is.null(st)) {
    st = log_env$mbspls_state %||% NULL
  }

  if (is.null(st)) {
    stop(sprintf("mbspls_state not found in %s.", where), call. = FALSE)
  }

  assert_mbspls_state(st, require_train_blocks = require_train_blocks, where = paste0(where, "$mbspls_state"))
  st
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

#' Create a backend primary-key column name that does not collide.
#' @keywords internal
mb_make_backend_key_name = function(existing, key_name = "..row_id") {
  key_name = key_name %||% "..row_id"
  if (!(key_name %in% existing)) {
    return(key_name)
  }
  make.unique(c(existing, key_name))[length(existing) + 1L]
}

# ------------------------------------------------------------------------------
# Multi-block task helpers
# ------------------------------------------------------------------------------

#' Normalize a multi-block mapping.
#' @keywords internal
mb_normalize_blocks = function(blocks, .var.name = "blocks") {
  checkmate::assert_list(
    blocks,
    types = "character",
    min.len = 1L,
    names = "unique",
    .var.name = .var.name
  )

  blocks = lapply(blocks, function(cols) unique(as.character(cols)))
  flat = unlist(blocks, use.names = FALSE)
  dup = unique(flat[duplicated(flat)])
  if (length(dup)) {
    stop(
      sprintf(
        "%s must be disjoint across blocks. Duplicated feature(s): %s",
        .var.name,
        paste(dup, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  blocks
}


#' Return TRUE only for numeric vectors with finite, positive variance.
#' @keywords internal
mb_has_finite_variance = function(x, tol = 1e-12) {
  if (!is.numeric(x)) {
    return(FALSE)
  }

  v = suppressWarnings(stats::var(x, na.rm = TRUE))
  is.finite(v) && !is.na(v) && v > tol
}


#' Expand stable base names to concrete task/backend column names.
#' @keywords internal
mb_expand_block_cols = function(dt_names, cols) {
  checkmate::assert_character(dt_names, any.missing = FALSE, .var.name = "dt_names")
  checkmate::assert_character(cols, any.missing = FALSE, min.len = 1L, .var.name = "cols")

  esc = function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\\\1", s)

  unique(unlist(lapply(cols, function(co) {
    if (co %in% dt_names) {
      co
    } else {
      grep(paste0("^", esc(co), "(\\\\.|$)"), dt_names, value = TRUE)
    }
  }), use.names = FALSE))
}


#' Resolve blocks against a concrete data table.
#' @keywords internal
mb_resolve_blocks = function(
  dt,
  blocks,
  numeric_only = TRUE,
  non_constant = TRUE) {

  if (is.null(blocks)) {
    return(NULL)
  }

  blocks = mb_normalize_blocks(blocks)
  dt = data.table::as.data.table(dt)
  dt_names = names(dt)

  out = lapply(blocks, function(cols) {
    cand = mb_expand_block_cols(dt_names, cols)

    if (isTRUE(numeric_only)) {
      cand = cand[vapply(cand, function(cl) is.numeric(dt[[cl]]), logical(1))]
    }
    if (!length(cand)) {
      return(character(0))
    }

    if (isTRUE(non_constant)) {
      cand = cand[vapply(cand, function(cl) mb_has_finite_variance(dt[[cl]]), logical(1))]
    }
    cand
  })

  Filter(length, out)
}


#' Extract block metadata from a multiblock task if available.
#' @keywords internal
mb_task_blocks = function(task, context = "task", allow_null = FALSE) {
  checkmate::assert_class(task, "Task", .var.name = paste0(context, "$task"))

  blocks = tryCatch(task$blocks, error = function(e) NULL)
  if (is.null(blocks)) {
    blocks = tryCatch(task$extra_args$blocks, error = function(e) NULL)
  }
  if (is.null(blocks)) {
    if (isTRUE(allow_null)) {
      return(NULL)
    }
    stop(
      sprintf(
        "%s: no 'blocks' supplied and the task does not carry multi-block metadata in `task$blocks` or `task$extra_args$blocks`.",
        context
      ),
      call. = FALSE
    )
  }

  mb_normalize_blocks(blocks, .var.name = paste0(context, "$blocks"))
}


#' Resolve a blocks argument for high-level graph constructors.
#' @keywords internal
mb_graph_blocks = function(blocks = NULL, task = NULL, context = "mbspls_graph") {
  if (!is.null(blocks)) {
    return(mb_normalize_blocks(blocks, .var.name = paste0(context, "$blocks")))
  }
  if (is.null(task)) {
    stop(
      sprintf("%s: supply either 'blocks' or a TaskMultiBlock via 'task'.", context),
      call. = FALSE
    )
  }
  mb_task_blocks(task, context = context)
}


#' Validate that referenced site-correction columns exist on a task.
#' @keywords internal
mb_validate_site_correction = function(task, site_correction = list(), context = "mbspls_graph") {
  if (is.null(task) || !length(site_correction)) {
    return(invisible(TRUE))
  }
  checkmate::assert_class(task, "Task", .var.name = paste0(context, "$task"))
  cols = unique(unlist(site_correction, recursive = TRUE, use.names = FALSE))
  cols = cols[nzchar(cols)]
  if (!length(cols)) {
    return(invisible(TRUE))
  }

  available = unique(c(
    task$feature_names,
    tryCatch(task$target_names, error = function(e) character(0))
  ))
  missing = setdiff(cols, available)
  if (length(missing)) {
    stop(
      sprintf(
        "%s: site-correction columns not found on the task: %s.",
        context,
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


#' Validate supervised TaskMultiBlock usage.
#' @keywords internal
mb_validate_supervised_task = function(task, context = "mbsplsxy_graph") {
  if (is.null(task)) {
    return(invisible(NULL))
  }
  checkmate::assert_class(task, "Task", .var.name = paste0(context, "$task"))
  task_type = tryCatch(task$task_type, error = function(e) NA_character_)
  if (!task_type %in% c("classif", "regr")) {
    stop(
      sprintf(
        "%s: supervised MB-sPLS-XY requires a classification or regression task, not '%s'.",
        context,
        as.character(task_type)
      ),
      call. = FALSE
    )
  }
  invisible(task_type)
}


#' Validate that a learner matches the expected supervised task type.
#' @keywords internal
mb_validate_supervised_learner = function(learner, expected_type, context = "mbsplsxy_graph_learner") {
  checkmate::assert_class(learner, "Learner", .var.name = paste0(context, "$learner"))
  checkmate::assert_choice(expected_type, c("classif", "regr"), .var.name = paste0(context, "$expected_type"))

  learner_type = tryCatch(learner$task_type, error = function(e) NA_character_)
  if (is.na(learner_type) || !nzchar(learner_type)) {
    return(invisible(TRUE))
  }
  if (!learner_type %in% c("classif", "regr")) {
    stop(
      sprintf(
        "%s: learner '%s' has task type '%s', but MB-sPLS-XY requires a classification or regression learner.",
        context,
        learner$id %||% "<unknown>",
        as.character(learner_type)
      ),
      call. = FALSE
    )
  }
  if (!identical(learner_type, expected_type)) {
    stop(
      sprintf(
        "%s: learner '%s' has task type '%s', which does not match the expected task type '%s'.",
        context,
        learner$id %||% "<unknown>",
        as.character(learner_type),
        expected_type
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}
