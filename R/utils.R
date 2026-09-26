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
#' Evaluates `fn()` after seeding the generator `kind` together with the
#' Inversion normal generator and the Rejection sampler, so a supplied seed
#' yields the same draws whatever RNG kinds the caller has selected. The
#' caller's RNG state is preserved: its RNG kinds and `.Random.seed` (including
#' the absence of `.Random.seed`) are restored on exit, also when `fn()` fails.
#' As with any seed-based restoration in R, a cached Box-Muller normal variate
#' of the calling session cannot be restored.
#'
#' With `seed = NULL`, `fn()` runs unseeded in the caller's RNG context and
#' nothing is changed or restored. `seed = 0` is an ordinary seed. The default
#' Mersenne-Twister generator reproduces results obtained under R's default RNG
#' kind. Use `kind = "L'Ecuyer-CMRG"` when `fn()` distributes work with
#' `parallel::mclapply()` or `parallel::mcparallel()` (`mc.set.seed = TRUE`):
#' these derive reproducible per-child streams only from that generator.
#'
#' This helper is intentionally implemented without additional dependencies
#' (e.g. withr) and is used to make bootstrap/permutation procedures reproducible
#' without permanently changing the session RNG state.
#'
#' @param seed `NULL` or one non-negative integer seed.
#' @param fn Function without arguments to evaluate.
#' @param kind RNG algorithm seeded for the evaluation: `"Mersenne-Twister"`
#'   (default) or `"L'Ecuyer-CMRG"`.
#' @return The value of `fn()`.
#' @keywords internal
with_seed_local = function(seed, fn, kind = c("Mersenne-Twister", "L'Ecuyer-CMRG")) {
  if (!is.function(fn)) {
    stop("`fn` must be a function.", call. = FALSE)
  }
  kind = match.arg(kind)
  if (is.null(seed)) {
    return(fn())
  }
  seed = .mb_assert_scalar_integer(seed, "seed", lower = 0L)

  old_kind = RNGkind()
  old_seed = if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit(.mb_restore_rng_state(old_kind, old_seed), add = TRUE)

  # Explicit algorithms make a supplied seed independent of ambient RNG kinds.
  # R cannot restore Box-Muller's cached spare normal via .Random.seed.
  set.seed(seed, kind = kind, normal.kind = "Inversion", sample.kind = "Rejection")
  fn()
}

# Restore RNG kinds and `.Random.seed` captured before a local RNG change;
# `seed = NULL` means the caller had no `.Random.seed`. Re-selecting a legacy
# kind (e.g. the "Rounding" sampler) makes R repeat the warning the caller
# already received when choosing it, so warnings of this call are muffled.
.mb_restore_rng_state = function(kind, seed) {
  withCallingHandlers(
    do.call(RNGkind, as.list(kind)),
    warning = function(w) invokeRestart("muffleWarning")
  )
  if (is.null(seed)) {
    if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  } else {
    assign(".Random.seed", seed, envir = .GlobalEnv)
  }
  invisible(NULL)
}

# Preserve the actual upstream topology, including single-node prefixes.
mb_preprocessing_graph = function(graph, target_id) {
  if (!target_id %in% graph$ids()) {
    stop("The requested component node is absent from the graph.", call. = FALSE)
  }
  edges = graph$edges
  ancestors = target_id
  repeat {
    expanded = union(ancestors, edges$src_id[edges$dst_id %in% ancestors])
    if (setequal(expanded, ancestors)) break
    ancestors = expanded
  }
  ancestors = setdiff(ancestors, target_id)
  result = mlr3pipelines::Graph$new()
  for (id in intersect(graph$ids(), ancestors)) {
    result$add_pipeop(graph$pipeops[[id]]$clone(deep = TRUE))
  }
  keep = edges$src_id %in% ancestors & edges$dst_id %in% ancestors
  for (i in which(keep)) {
    result$add_edge(
      edges$src_id[[i]], edges$dst_id[[i]],
      edges$src_channel[[i]], edges$dst_channel[[i]]
    )
  }
  result
}

mb_assert_resampling_split = function(task, train, test) {
  if (!length(train) || !length(test) ||
    anyNA(c(train, test)) ||
    !all(c(train, test) %in% task$row_ids)) {
    stop("Resampling indices must select non-empty subsets of the current task.",
      call. = FALSE)
  }
  if (length(intersect(train, test))) {
    stop("Resampling analysis and assessment rows must be disjoint.", call. = FALSE)
  }
  train_groups = mb_task_group_vector(task, train)
  test_groups = mb_task_group_vector(task, test)
  if (!is.null(train_groups) || !is.null(test_groups)) {
    mb_assert_disjoint_groups(train_groups, test_groups)
  }
  invisible(TRUE)
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

# Look up the prediction payload that belongs to one fitted MB-sPLS/MB-sPCA
# node of a trained GraphLearner.
#
# Payloads are stored per training run in `log_env$mbspls_last[[run_id]]`.
# The run id is read from the fitted PipeOp state `learner$model[[pipeop_id]]`
# (a fitted PipeOp object is accepted as well) and the log environment from
# the PipeOp's parameters, so each resampling iteration, benchmarked learner or
# tuning configuration is scored on its own prediction. When a run id is known
# but no payload is stored for it (e.g. the prediction ran in a parallel worker
# whose `log_env` never reached this process), NULL is returned so the measure
# becomes NA. The shared `log_env$last` is used, with a warning, only for a
# fitted state that records no run id at all.
.mb_prediction_payload = function(learner, pipeop_id) {
  fit = tryCatch(learner$model[[pipeop_id]], error = function(e) NULL)
  env = NULL
  if (inherits(fit, "PipeOp")) {
    env = tryCatch(fit$param_set$values$log_env, error = function(e) NULL)
    fit = fit$state
  }
  if (!inherits(env, "environment")) {
    env = tryCatch(learner$graph$pipeops[[pipeop_id]]$param_set$values$log_env,
      error = function(e) NULL)
  }
  if (!inherits(env, "environment") || !is.list(fit) || !length(fit)) {
    return(NULL)
  }

  run_id = fit[["run_id"]]
  if (is.character(run_id) && length(run_id) == 1L && !is.na(run_id) && nzchar(run_id)) {
    payload = env$mbspls_last[[run_id]]
    return(if (is.list(payload)) payload else NULL)
  }

  if (!is.list(env$last)) {
    return(NULL)
  }
  warning(
    sprintf(
      paste0(
        "The fitted state of PipeOp '%s' records no run id; using the most recent ",
        "prediction payload in log_env$last, which may belong to another ",
        "resampling iteration or learner."
      ),
      pipeop_id
    ),
    call. = FALSE
  )
  env$last
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

  # loadings must also be present (required for deflation in prediction)
  if (!is.null(st$loadings) && length(st$loadings)) {
    checkmate::assert_list(st$loadings, names = "strict", .var.name = paste0(where, "$loadings"))
  }

  # optional but common
  if (!is.null(st$ncomp)) {
    checkmate::assert_integerish(st$ncomp, len = 1L, lower = 0L, .var.name = paste0(where, "$ncomp"))

    # ev_comp must have exactly ncomp entries if present
    if (!is.null(st$ev_comp) && st$ncomp > 0L) {
      checkmate::assert_numeric(st$ev_comp, len = st$ncomp,
        .var.name = paste0(where, "$ev_comp"))
    }

    # ev_block must have ncomp rows and one column per block if present
    B = length(st$blocks)
    if (!is.null(st$ev_block) && st$ncomp > 0L && B > 0L) {
      checkmate::assert_matrix(st$ev_block, nrows = st$ncomp, ncols = B,
        .var.name = paste0(where, "$ev_block"))
    }
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


#' Resolve declared block columns against concrete data column names.
#'
#' Maps a block mapping declared on stable (pre-encoding) feature names to the
#' columns of a concrete data table. This is the single rule set for resolving
#' blocks after upstream preprocessing.
#'
#' @details
#' Resolution rules:
#'
#' 1. A declared name present in `dt_names` maps exactly to itself.
#' 2. A declared name absent from `dt_names`, typically a factor replaced by
#'    encoded columns such as `sex.m`, expands to the columns starting with
#'    `paste0(name, ".")`, kept in data order. A column equal to a declared name
#'    of any block is never claimed by such an expansion.
#' 3. A column matching several absent names is assigned to the longest one,
#'    e.g. `sex.hormone.high` belongs to `sex.hormone`, not to `sex`.
#' 4. The resolved blocks must be disjoint; a column resolved into more than
#'    one block is an error.
#'
#' Prefixes are matched literally, so names containing regular-expression
#' metacharacters need no escaping. Expansion cannot distinguish encoder output
#' from an undeclared column that happens to start with `<name>.`; declare such
#' columns in their own block to keep them out of other blocks.
#'
#' @param dt_names Character vector of data column names.
#' @param blocks Named list of declared column names per block.
#' @return Named list of resolved column vectors in block order. Blocks without
#'   any matching column yield `character(0)`.
#' @keywords internal
mb_resolve_block_columns = function(dt_names, blocks) {
  checkmate::assert_character(dt_names, any.missing = FALSE, .var.name = "dt_names")
  checkmate::assert_list(blocks, types = "character", min.len = 1L, names = "unique",
    .var.name = "blocks")

  blocks = lapply(blocks, function(cols) unique(as.character(cols)))
  declared = unique(unlist(blocks, use.names = FALSE))
  if (anyNA(declared)) {
    stop("`blocks` must not contain missing column names.", call. = FALSE)
  }
  absent = setdiff(declared, dt_names)

  # Candidates for prefix expansion never include a declared name. Each one is
  # owned by the longest absent declared name that is a prefix of it.
  candidates = dt_names[!dt_names %in% declared]
  owner = rep(NA_character_, length(candidates))
  if (length(absent) && length(candidates)) {
    owner_nchar = integer(length(candidates))
    for (base in absent) {
      hit = startsWith(candidates, paste0(base, ".")) & nchar(base) > owner_nchar
      owner[hit] = base
      owner_nchar[hit] = nchar(base)
    }
  }

  resolved = lapply(blocks, function(cols) {
    unique(as.character(unlist(lapply(cols, function(co) {
      if (co %in% dt_names) co else candidates[!is.na(owner) & owner == co]
    }), use.names = FALSE)))
  })

  flat = unlist(resolved, use.names = FALSE)
  shared = unique(flat[duplicated(flat)])
  if (length(shared)) {
    block_of = rep(names(resolved), lengths(resolved))
    detail = vapply(shared, function(cl) {
      sprintf("%s (%s)", cl, paste(block_of[flat == cl], collapse = ", "))
    }, character(1L))
    stop(
      sprintf(
        "Resolved block columns must be disjoint across blocks. Column(s) assigned to several blocks: %s",
        mb_format_truncated(detail)
      ),
      call. = FALSE
    )
  }

  resolved
}


#' Expand the declared names of one block to concrete data column names.
#'
#' Applies the rules of [mb_resolve_block_columns()] to a single block. Pass the
#' declared names of all blocks as `declared`, so that expansion never claims a
#' column declared in another block and competing absent names are assigned as
#' in the full mapping.
#'
#' @param dt_names Character vector of data column names.
#' @param cols Declared column names of the block.
#' @param declared Declared column names of all blocks. Defaults to `cols`.
#' @return Character vector of resolved column names.
#' @keywords internal
mb_expand_block_cols = function(dt_names, cols, declared = cols) {
  checkmate::assert_character(cols, any.missing = FALSE, min.len = 1L, .var.name = "cols")
  checkmate::assert_character(declared, any.missing = FALSE, .var.name = "declared")

  cols = unique(cols)
  blocks = list(block = cols)
  others = setdiff(declared, cols)
  if (length(others)) {
    blocks$other = others
  }
  mb_resolve_block_columns(dt_names, blocks)[["block"]]
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
  resolved = mb_resolve_block_columns(names(dt), blocks)

  out = lapply(names(blocks), function(bn) {
    cols = blocks[[bn]]
    cand = resolved[[bn]]

    if (isTRUE(numeric_only)) {
      cand = cand[vapply(cand, function(cl) is.numeric(dt[[cl]]), logical(1))]
    }
    if (!length(cand)) {
      warning(sprintf(
        "mb_resolve_blocks: block '%s' matched 0 columns in the data (requested: %s). Check that column names and any encoding suffixes match exactly.",
        bn, mb_format_truncated(cols)
      ), call. = FALSE)
      return(character(0))
    }

    if (isTRUE(non_constant)) {
      cand = cand[vapply(cand, function(cl) mb_has_finite_variance(dt[[cl]]), logical(1))]
    }
    cand
  })
  names(out) = names(blocks)

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


# ------------------------------------------------------------------------------
# Suggested packages
# ------------------------------------------------------------------------------

# Thin wrapper around requireNamespace() so that tests can simulate a missing
# suggested package.
.mbspls_has_namespace = function(pkg) {
  requireNamespace(pkg, quietly = TRUE)
}

# Stop with an installation hint unless every suggested package in `pkgs` can be
# loaded. `what` names the feature that needs them, e.g. "autoplot(type = 'x')".
.mbspls_require_suggested = function(pkgs, what) {
  checkmate::assert_character(pkgs, any.missing = FALSE, min.len = 1L, .var.name = "pkgs")
  checkmate::assert_string(what, .var.name = "what")

  missing = pkgs[!vapply(pkgs, .mbspls_has_namespace, logical(1L))]
  if (length(missing)) {
    stop(
      sprintf(
        "%s requires the suggested package(s) %s. Install with install.packages(c(%s)).",
        what,
        paste(sprintf("'%s'", missing), collapse = ", "),
        paste(sprintf("\"%s\"", missing), collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}
