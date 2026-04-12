mb_drop_target_from_blocks = function(blocks, target = NULL, context = "TaskMultiBlock()") {
  blocks = mb_normalize_blocks(blocks, .var.name = paste0(context, "$blocks"))
  if (is.null(target) || !length(target) || !nzchar(as.character(target)[1L])) {
    return(blocks)
  }

  target = as.character(target)[1L]
  out = lapply(blocks, function(cols) setdiff(cols, target))
  empty = names(out)[lengths(out) == 0L]
  if (length(empty)) {
    stop(
      sprintf(
        "%s: the target column '%s' is part of block(s) %s, leaving them empty after removing the target.\nFix: keep the target outside the block mapping or add at least one non-target feature to each affected block.",
        context,
        target,
        mb_format_truncated(empty)
      ),
      call. = FALSE
    )
  }

  out
}


mb_blocks_rename = function(blocks, old, new) {
  blocks = mb_normalize_blocks(blocks, .var.name = "blocks")
  checkmate::assert_character(old, any.missing = FALSE, min.len = 1L, .var.name = "old")
  checkmate::assert_character(new, any.missing = FALSE, len = length(old), .var.name = "new")

  lookup = stats::setNames(as.character(new), as.character(old))
  lapply(blocks, function(cols) {
    cols = as.character(cols)
    idx = match(cols, names(lookup), nomatch = 0L)
    if (any(idx > 0L)) {
      cols[idx > 0L] = unname(lookup[cols[idx > 0L]])
    }
    cols
  })
}


mb_task_clone_backend = function(task, context = "as_task_multiblock()") {
  checkmate::assert_class(task, "Task", .var.name = paste0(context, "$task"))
  backend = tryCatch(task$backend, error = function(e) NULL)
  if (is.null(backend)) {
    stop(sprintf("%s: could not access the task backend.", context), call. = FALSE)
  }

  out = tryCatch(backend$clone(deep = TRUE), error = function(e) NULL)
  if (is.null(out)) {
    out = tryCatch(backend$clone(), error = function(e) NULL)
  }
  out %||% backend
}


mb_task_resolve_constructor_blocks = function(blocks = NULL, extra_args = list(), context = "TaskMultiBlock") {
  extra_args = extra_args %||% list()
  checkmate::assert_list(extra_args, .var.name = paste0(context, "$extra_args"))

  resolved = blocks %||% extra_args$blocks %||% NULL
  if (is.null(resolved)) {
    stop(
      sprintf(
        "%s: missing multi-block metadata. Supply `blocks` explicitly or pass them via `extra_args$blocks`.",
        context
      ),
      call. = FALSE
    )
  }

  mb_normalize_blocks(resolved, .var.name = paste0(context, "$blocks"))
}


mb_task_copy_view = function(dst, src) {
  checkmate::assert_class(dst, "Task", .var.name = "dst")
  checkmate::assert_class(src, "Task", .var.name = "src")

  try(dst$row_roles <- src$row_roles, silent = TRUE)
  try(dst$col_roles <- src$col_roles, silent = TRUE)
  try(dst$col_labels <- src$col_labels, silent = TRUE)

  internal_valid = tryCatch(src$internal_valid_task, error = function(e) NULL)
  if (!is.null(internal_valid)) {
    try(dst$internal_valid_task <- internal_valid, silent = TRUE)
  }

  invisible(dst)
}


mb_task_construct_from_source = function(
  source_task,
  blocks,
  target,
  resolved_type,
  id,
  label,
  positive = NULL
) {
  checkmate::assert_class(source_task, "Task", .var.name = "source_task")
  checkmate::assert_choice(resolved_type, c("classif", "regr", "clust"), .var.name = "resolved_type")

  source_copy = source_task$clone(deep = TRUE)
  source_target = tryCatch(source_copy$target_names, error = function(e) character(0))
  source_target = source_target[1L] %||% NULL
  target_col = if (identical(resolved_type, "clust")) NULL else (target %||% source_target)

  if (!identical(resolved_type, "clust") && is.null(target_col)) {
    stop(
      sprintf(
        "as_task_multiblock(): task_type = '%s' requires a target column.",
        resolved_type
      ),
      call. = FALSE
    )
  }

  blocks = mb_drop_target_from_blocks(blocks, target_col, context = "as_task_multiblock()")

  if (!identical(source_copy$task_type, resolved_type) || !identical(target_col, source_target)) {
    source_copy = mlr3::convert_task(
      source_copy,
      target = target_col,
      new_type = resolved_type,
      drop_original_target = FALSE
    )
  }

  backend = mb_task_clone_backend(source_copy, context = "as_task_multiblock()")
  ctor_extra_args = list(blocks = blocks)
  ctor_args = list(
    id = id,
    backend = backend,
    blocks = blocks,
    label = label,
    extra_args = ctor_extra_args
  )

  task = switch(
    resolved_type,
    classif = {
      ctor_args$target = target_col
      ctor_args$positive = positive
      do.call(TaskClassifMultiBlock$new, ctor_args)
    },
    regr = {
      ctor_args$target = target_col
      do.call(TaskRegrMultiBlock$new, ctor_args)
    },
    clust = do.call(TaskClustMultiBlock$new, ctor_args)
  )

  mb_task_copy_view(task, source_copy)

  if (identical(resolved_type, "classif") && is.null(positive) && inherits(source_task, "TaskClassif")) {
    pos = tryCatch(source_task$positive, error = function(e) NA_character_)
    if (!is.na(pos) && pos %in% task$class_names) {
      task$positive = pos
    }
  }

  task
}


#' Multi-block task factory, packaged toy tasks, and dataset adapters
#'
#' @title Construct multi-block mlr3 tasks with optional supervision
#'
#' @description
#' `TaskMultiBlock()` is a factory that returns the appropriate mlr3 task type
#' for a multi-block analysis:
#'
#' * [mlr3cluster::TaskClust] subclass for unsupervised use (`task_type = "clust"`),
#' * [mlr3::TaskClassif] subclass for supervised classification,
#' * [mlr3::TaskRegr] subclass for supervised regression.
#'
#' The underlying task always stores a named `blocks` mapping as task metadata
#' (including `extra_args` for task conversion) and exposes three convenience
#' methods:
#'
#' * `$block_features(block = NULL, materialize = FALSE)`
#' * `$block_data(rows = NULL, blocks = NULL, as_matrix = FALSE)`
#' * `$overview(rows = NULL, blocks = NULL, include_target = TRUE)`
#'
#' This design follows the mlr3 task hierarchy: supervised and unsupervised
#' tasks live on different branches, so the package uses a factory plus thin
#' task-type-specific subclasses instead of one monolithic class.
#'
#' `as_task_multiblock()` accepts either a flat backend plus a block mapping, or
#' a named list of aligned blocks which are flattened automatically with stable
#' `<block><sep><feature>` column names.
#'
#' `task_multiblock_synthetic()` provides a packaged 3-block toy data generator
#' for unsupervised, classification, and regression examples.
#'
#' `task_multiblock_breast_tcga()` and `task_multiblock_potato()` are optional
#' dataset adapters for the `mixOmics::breast.TCGA` and `multiblock::potato`
#' examples discussed in the package README.
#'
#' @param x A flat tabular object (`data.frame`, `data.table`, `matrix`,
#'   `mlr3::DataBackend`, or `mlr3::Task`) or a named `list` of aligned blocks.
#' @param blocks Named list mapping block names to feature columns. Required for
#'   flat input. Ignored for list-of-block input because the mapping is created
#'   automatically.
#' @param target Optional target. For flat input this can be a target column name
#'   or a target vector. For list-of-block input this can be `NULL` or a target
#'   vector. When `x` is already an `mlr3::Task`, `target` must be `NULL` or the
#'   name of an existing backend column so that the original task backend and
#'   view can be preserved.
#' @param task_type One of `"auto"`, `"classif"`, `"regr"`, or `"clust"`.
#'   With `"auto"`, the factory infers the task type from `target`: factor-like
#'   targets become classification tasks, numeric targets become regression
#'   tasks, and missing targets yield clustering tasks.
#' @param id Task id.
#' @param target_name Name to use if a target vector must be appended to the
#'   backend.
#' @param positive Optional positive class passed through to classification
#'   tasks.
#' @param label Optional task label.
#' @param sep Separator used when flattening list-of-block input.
#' @param n Number of rows for the synthetic task.
#' @param seed RNG seed for the synthetic task.
#' @param subset Which subset of `mixOmics::breast.TCGA` to use. Defaults to the
#'   training subset because the packaged test subset omits the protein block.
#' @param response Which sensory response to use for the potato regression task.
#'   Either a column index or column name.
#' @param ... Passed through to [as_task_multiblock()].
#' @return A multiblock task that inherits from the corresponding mlr3 task
#'   type and carries a named `blocks` mapping.
#'
#' @examples
#' blocks = list(
#'   clin = c("clin_age", "clin_bmi"),
#'   geno = c("geno_1", "geno_2", "geno_3"),
#'   prot = c("prot_1", "prot_2")
#' )
#'
#' dt = data.table::data.table(
#'   clin_age = rnorm(20),
#'   clin_bmi = rnorm(20),
#'   geno_1 = rnorm(20),
#'   geno_2 = rnorm(20),
#'   geno_3 = rnorm(20),
#'   prot_1 = rnorm(20),
#'   prot_2 = rnorm(20),
#'   y = factor(sample(c("a", "b"), 20, TRUE))
#' )
#'
#' task = TaskMultiBlock(dt, blocks = blocks, target = "y")
#' task$block_names
#' names(task$block_data(as_matrix = TRUE))
#' task$overview()$blocks
#'
#' toy_unsup = task_multiblock_synthetic(task_type = "clust")
#' toy_class = task_multiblock_synthetic(task_type = "classif")
#' toy_regr = task_multiblock_synthetic(task_type = "regr")
#'
#' \dontrun{
#' if (requireNamespace("mixOmics", quietly = TRUE)) {
#'   tcga = try(task_multiblock_breast_tcga(task_type = "classif"), silent = TRUE)
#'   if (!inherits(tcga, "try-error")) tcga$block_names
#' }
#'
#' if (requireNamespace("multiblock", quietly = TRUE)) {
#'   pot = task_multiblock_potato(task_type = "regr")
#'   pot$block_names
#' }
#' }
#'

#' @export
as_task_multiblock = function(
  x,
  blocks = NULL,
  target = NULL,
  task_type = c("auto", "classif", "regr", "clust"),
  id = NULL,
  target_name = NULL,
  positive = NULL,
  label = NA_character_,
  sep = "__") {

  task_type = match.arg(task_type)
  source_task = NULL

  if (inherits(x, "Task")) {
    source_task = x
    id = id %||% x$id
    if (identical(label, NA_character_)) {
      label = x$label
    }
    if (is.null(blocks)) {
      blocks = tryCatch(x$blocks, error = function(e) NULL)
      if (is.null(blocks)) {
        blocks = tryCatch(x$extra_args$blocks, error = function(e) NULL)
      }
    }
    if (is.null(target) && (inherits(x, "TaskClassif") || inherits(x, "TaskRegr"))) {
      target = x$target_names[1L]
    }
    if (is.null(blocks)) {
      stop(
        paste0(
          "as_task_multiblock(): no multi-block metadata was found on the supplied task.",
          "\nFix: pass an explicit named `blocks` mapping, or construct the source task with `TaskMultiBlock()` so that `task$blocks` / `task$extra_args$blocks` is available."
        ),
        call. = FALSE
      )
    }
    if (!is.null(target) && !(length(target) == 1L && is.character(target) && target %in% x$col_info$id)) {
      stop(
        paste0("When `x` is an mlr3 Task, `target` must be NULL or the name of an existing backend column.",
          "\nFix: for an external target vector, call `TaskMultiBlock()` on raw tabular or list-of-block input instead of an existing Task so that the original task backend, view, and metadata remain intact."),
        call. = FALSE
      )
    }

    resolved_type = mb_infer_task_type(
      task_type = task_type,
      dt = NULL,
      target = target,
      source_task = source_task
    )

    return(mb_task_construct_from_source(
      source_task = source_task,
      blocks = blocks,
      target = target,
      resolved_type = resolved_type,
      id = id,
      label = label,
      positive = positive
    ))
  }

  id = id %||% "multiblock"

  prep = mb_prepare_multiblock_input(
    x = x,
    blocks = blocks,
    target = target,
    target_name = target_name,
    sep = sep
  )

  resolved_type = mb_infer_task_type(
    task_type = task_type,
    dt = prep$backend,
    target = prep$target,
    source_task = NULL
  )

  if (!identical(resolved_type, "clust") && is.null(prep$target)) {
    stop(
      sprintf("TaskMultiBlock(): task_type = '%s' requires a target column or target vector.", resolved_type),
      call. = FALSE
    )
  }

  prep$backend = mb_finalize_target_type(
    dt = prep$backend,
    target = prep$target,
    task_type = resolved_type
  )

  target_col = if (identical(resolved_type, "clust")) NULL else prep$target

  switch(
    resolved_type,
    classif = TaskClassifMultiBlock$new(
      id = id,
      backend = prep$backend,
      target = target_col,
      blocks = prep$blocks,
      positive = positive,
      label = label,
      extra_args = list(blocks = prep$blocks)
    ),
    regr = TaskRegrMultiBlock$new(
      id = id,
      backend = prep$backend,
      target = target_col,
      blocks = prep$blocks,
      label = label,
      extra_args = list(blocks = prep$blocks)
    ),
    clust = TaskClustMultiBlock$new(
      id = id,
      backend = prep$backend,
      blocks = prep$blocks,
      label = label,
      extra_args = list(blocks = prep$blocks)
    )
  )
}


#' @rdname as_task_multiblock
#' @export
TaskMultiBlock = function(...) {
  as_task_multiblock(...)
}


#' @rdname as_task_multiblock
#' @export
task_multiblock_synthetic = function(
  task_type = c("clust", "classif", "regr"),
  id = NULL,
  n = 90L,
  seed = 1L,
  label = NA_character_) {

  task_type = match.arg(task_type)
  dat = mb_generate_synthetic_multiblock(n = n, seed = seed)

  features = data.table::copy(dat$dt)
  features[, c("subtype", "response") := NULL]

  target = switch(
    task_type,
    clust = NULL,
    classif = dat$dt$subtype,
    regr = dat$dt$response
  )

  if (is.null(id)) {
    id = switch(
      task_type,
      clust = "mbspls_synthetic_blocks",
      classif = "mbspls_synthetic_classif",
      regr = "mbspls_synthetic_regr"
    )
  }

  if (identical(label, NA_character_)) {
    label = switch(
      task_type,
      clust = "Synthetic 3-block MB-sPLS clustering task",
      classif = "Synthetic 3-block MB-sPLS classification task",
      regr = "Synthetic 3-block MB-sPLS regression task"
    )
  }

  as_task_multiblock(
    x = features,
    blocks = dat$blocks,
    target = target,
    task_type = task_type,
    id = id,
    target_name = if (identical(task_type, "classif")) "subtype" else "response",
    label = label
  )
}


#' @rdname as_task_multiblock
#' @export
task_multiblock_breast_tcga = function(
  task_type = c("classif", "clust"),
  id = NULL,
  subset = c("train"),
  label = NA_character_) {

  task_type = match.arg(task_type)
  subset = match.arg(subset)

  if (!requireNamespace("mixOmics", quietly = TRUE)) {
    stop(
      "Package 'mixOmics' is required for `task_multiblock_breast_tcga()`. ",
      "Install it first and try again.",
      call. = FALSE
    )
  }

  utils::data("breast.TCGA", package = "mixOmics", envir = environment())
  tcga = get("breast.TCGA", envir = environment())
  ds = tcga[[paste0("data.", subset)]]
  block_candidates = c("mRNA", "miRNA", "protein")
  blocks_present = Filter(
    function(nm) {
      obj = ds[[nm]]
      !is.null(obj) && isTRUE(length(dim(obj)) == 2L)
    },
    block_candidates
  )
  if (length(blocks_present) < 2L) {
    stop(
      "mixOmics::breast.TCGA does not provide enough 2D omics blocks in this installation.",
      call. = FALSE
    )
  }

  if (is.null(id)) {
    id = if (identical(task_type, "classif")) "breast_tcga_multiblock" else "breast_tcga_multiblock_clust"
  }
  if (identical(label, NA_character_)) {
    label = if (identical(task_type, "classif")) {
      "TCGA breast cancer 3-block classification task"
    } else {
      "TCGA breast cancer 3-block unsupervised task"
    }
  }

  TaskMultiBlock(
    x = ds[blocks_present],
    target = if (identical(task_type, "classif")) ds$subtype else NULL,
    task_type = task_type,
    id = id,
    target_name = "subtype",
    label = label
  )
}


#' @rdname as_task_multiblock
#' @export
task_multiblock_potato = function(
  task_type = c("regr", "clust"),
  id = NULL,
  response = 1L,
  label = NA_character_) {

  task_type = match.arg(task_type)

  if (!requireNamespace("multiblock", quietly = TRUE)) {
    stop(
      "Package 'multiblock' is required for `task_multiblock_potato()`. ",
      "Install it first and try again.",
      call. = FALSE
    )
  }

  utils::data("potato", package = "multiblock", envir = environment())
  potato = get("potato", envir = environment())

  sensory = potato$Sensory
  y = NULL
  if (identical(task_type, "regr")) {
    if (is.character(response)) {
      if (!response %in% colnames(sensory)) {
        stop(sprintf("Unknown potato response '%s'.", response), call. = FALSE)
      }
      y = sensory[[response]]
    } else {
      response = as.integer(response)
      if (response < 1L || response > ncol(sensory)) {
        stop("`response` is out of bounds for potato$Sensory.", call. = FALSE)
      }
      y = sensory[[response]]
    }
  }

  if (is.null(id)) {
    id = if (identical(task_type, "regr")) "potato_multiblock_regr" else "potato_multiblock_clust"
  }
  if (identical(label, NA_character_)) {
    label = if (identical(task_type, "regr")) {
      "Potato 3-block regression task"
    } else {
      "Potato 3-block unsupervised task"
    }
  }

  TaskMultiBlock(
    x = potato[c("Chemical", "Compression", "NIRraw")],
    target = y,
    task_type = task_type,
    id = id,
    target_name = "y",
    label = label
  )
}


mb_backend_as_data_table = function(x) {
  if (inherits(x, "DataBackend")) {
    cols = x$col_info$id
    return(data.table::as.data.table(x$data(cols = cols)))
  }
  if (data.table::is.data.table(x)) {
    return(data.table::copy(x))
  }
  if (is.matrix(x)) {
    x = as.data.frame(x, stringsAsFactors = FALSE)
  }
  data.table::as.data.table(x)
}


mb_make_target_name = function(existing, target_name = ".target") {
  target_name = target_name %||% ".target"
  if (!(target_name %in% existing)) {
    return(target_name)
  }
  make.unique(c(existing, target_name))[length(existing) + 1L]
}


mb_prepare_multiblock_input = function(x, blocks = NULL, target = NULL, target_name = NULL, sep = "__") {
  if (is.list(x) && !inherits(x, "data.frame") && !data.table::is.data.table(x)) {
    if (!length(x)) {
      stop("List input for `TaskMultiBlock()` must contain at least one block.", call. = FALSE)
    }
    if (is.null(names(x)) || any(!nzchar(names(x)))) {
      names(x) = paste0("block_", seq_along(x))
    }

    n_rows = unique(vapply(x, nrow, integer(1)))
    if (length(n_rows) != 1L) {
      stop("All blocks must have the same number of rows.", call. = FALSE)
    }

    pieces = Map(function(xb, bn) {
      xb = as.data.frame(xb, stringsAsFactors = FALSE)
      if (is.null(colnames(xb))) {
        colnames(xb) = paste0("V", seq_len(ncol(xb)))
      }
      colnames(xb) = paste0(bn, sep, make.names(colnames(xb), unique = TRUE))
      data.table::as.data.table(xb)
    }, x, names(x))

    dt = pieces[[1L]]
    if (length(pieces) > 1L) {
      for (i in 2:length(pieces)) {
        dt = cbind(dt, pieces[[i]])
      }
    }

    block_map = lapply(pieces, names)
    names(block_map) = names(pieces)

    target_col = NULL
    if (!is.null(target)) {
      if (length(target) == 1L && is.character(target) && target %in% names(dt)) {
        target_col = target
      } else {
        if (length(target) != n_rows) {
          stop("Target vector must have the same number of rows as the blocks.", call. = FALSE)
        }
        target_col = mb_make_target_name(names(dt), target_name)
        dt[[target_col]] = target
      }
    }

    block_map = mb_drop_target_from_blocks(block_map, target_col, context = "TaskMultiBlock()")
    assert_blocks_present(names(dt), block_map, context = "TaskMultiBlock()")
    return(list(backend = dt, blocks = block_map, target = target_col))
  }

  dt = mb_backend_as_data_table(x)
  if (is.null(blocks)) {
    stop("Flat input requires an explicit named `blocks` mapping.", call. = FALSE)
  }

  block_map = mb_normalize_blocks(blocks)
  target_col = NULL

  if (!is.null(target)) {
    if (length(target) == 1L && is.character(target) && target %in% names(dt)) {
      target_col = target
    } else {
      if (length(target) != nrow(dt)) {
        stop("Target vector must have the same number of rows as the backend.", call. = FALSE)
      }
      target_col = mb_make_target_name(names(dt), target_name)
      dt[[target_col]] = target
    }
  }

  block_map = mb_drop_target_from_blocks(block_map, target_col, context = "TaskMultiBlock()")
  assert_blocks_present(names(dt), block_map, context = "TaskMultiBlock()")

  list(backend = dt, blocks = block_map, target = target_col)
}


mb_infer_task_type = function(task_type, dt, target, source_task = NULL) {
  if (!identical(task_type, "auto")) {
    return(task_type)
  }

  if (!is.null(target)) {
    if (is.null(dt)) {
      if (is.null(source_task)) {
        stop("Could not infer `task_type`: no data available for the supplied target.", call. = FALSE)
      }
      y = data.table::as.data.table(source_task$backend$data(rows = source_task$row_ids, cols = target))[[1L]]
    } else {
      y = dt[[target]]
    }

    if (is.factor(y) || is.character(y) || is.logical(y) || is.ordered(y)) {
      return("classif")
    }
    if (is.numeric(y) || is.integer(y)) {
      return("regr")
    }

    stop("Could not infer `task_type` from the supplied target.", call. = FALSE)
  }

  if (!is.null(source_task)) {
    if (inherits(source_task, "TaskClassif")) {
      return("classif")
    }
    if (inherits(source_task, "TaskRegr")) {
      return("regr")
    }
    if (inherits(source_task, "TaskClust")) {
      return("clust")
    }
  }

  "clust"
}


mb_finalize_target_type = function(dt, target, task_type) {
  if (is.null(target)) {
    return(dt)
  }

  dt = data.table::copy(data.table::as.data.table(dt))

  if (identical(task_type, "classif")) {
    dt[[target]] = as.factor(dt[[target]])
    return(dt)
  }

  if (identical(task_type, "regr")) {
    y = dt[[target]]
    if (!(is.numeric(y) || is.integer(y))) {
      stop("For `task_type = 'regr'`, the target must be numeric or integer.", call. = FALSE)
    }
    dt[[target]] = as.numeric(y)
    return(dt)
  }

  if (target %in% names(dt)) {
    dt[, (target) := NULL]
  }
  dt
}


mb_task_block_features = function(task, block = NULL, materialize = FALSE) {
  blocks = mb_task_blocks(task, context = class(task)[1L])
  if (isTRUE(materialize)) {
    blocks = lapply(blocks, function(cols) {
      intersect(mb_expand_block_cols(task$feature_names, cols), task$feature_names)
    })
    blocks = Filter(length, blocks)
  }

  if (is.null(block)) {
    return(blocks)
  }

  checkmate::assert_choice(block, names(blocks), .var.name = "block")
  blocks[[block]]
}


mb_task_block_data = function(task, rows = NULL, blocks = NULL, as_matrix = FALSE) {
  rows = rows %||% task$row_ids

  blocks_map = if (is.null(blocks)) {
    mb_task_block_features(task, materialize = TRUE)
  } else if (is.character(blocks) && is.null(names(blocks))) {
    available = mb_task_block_features(task, materialize = TRUE)
    checkmate::assert_subset(blocks, names(available), .var.name = "blocks")
    available[blocks]
  } else {
    dt_task = data.table::as.data.table(task$data(rows = rows, cols = task$feature_names))
    requested = mb_normalize_blocks(blocks, .var.name = "blocks")
    resolved = mb_resolve_blocks(dt_task, requested, numeric_only = FALSE, non_constant = FALSE)
    empty_blocks = setdiff(names(requested), names(resolved))
    if (length(empty_blocks)) {
      stop(
        sprintf(
          "`task$block_data()` could not resolve the requested block mapping against the task's active feature view for block(s): %s
Fix: use block names from `task$block_names` / `task$block_features(materialize = TRUE)`, or reselect the required features before extracting block data.",
          mb_format_truncated(empty_blocks)
        ),
        call. = FALSE
      )
    }
    resolved
  }

  cols = unique(unlist(blocks_map, use.names = FALSE))
  dt = if (length(cols)) {
    data.table::as.data.table(task$data(rows = rows, cols = cols))
  } else {
    data.table::data.table()
  }

  out = lapply(blocks_map, function(cols_block) {
    x_block = dt[, ..cols_block]
    if (isTRUE(as_matrix)) {
      storage_ok = vapply(cols_block, function(cl) {
        is.numeric(x_block[[cl]]) || is.integer(x_block[[cl]]) || is.logical(x_block[[cl]])
      }, logical(1))
      if (!all(storage_ok)) {
        bad = cols_block[!storage_ok]
        stop(
          sprintf(
            paste0("`task$block_data(as_matrix = TRUE)` requires numeric, integer, or logical features only. Block columns with unsupported storage types: %s",
              "\nFix: encode factor/character features before requesting matrix output, or call `task$block_data(as_matrix = FALSE)`."),
            mb_format_truncated(bad)
          ),
          call. = FALSE
        )
      }
      data.matrix(x_block)
    } else {
      x_block
    }
  })

  names(out) = names(blocks_map)
  out
}


mb_generate_synthetic_multiblock = function(n = 90L, seed = 1L) {
  n = as.integer(n)
  seed = as.integer(seed)

  with_seed_local(seed, function() {
    site_batch = factor(sample(c("A", "B", "C"), size = n, replace = TRUE))
    z1 = stats::rnorm(n)
    z2 = stats::rnorm(n)
    z3 = stats::rnorm(n)
    site_shift = c(A = 0.0, B = 0.6, C = -0.6)[as.character(site_batch)]

    make_block = function(prefix, loads, noise_sd = 0.35, site_scale = 0.2) {
      X = vapply(seq_len(nrow(loads)), function(j) {
        loads[j, 1L] * z1 +
          loads[j, 2L] * z2 +
          loads[j, 3L] * z3 +
          site_scale * site_shift +
          stats::rnorm(n, sd = noise_sd)
      }, numeric(n))

      X = data.table::as.data.table(X)
      data.table::setnames(X, paste0(prefix, "_", seq_len(ncol(X))))
      X
    }

    block_a = make_block(
      "block_a",
      matrix(c(
        1.2, 0.4, 0.0,
        1.0, 0.3, 0.2,
        0.9, 0.5, 0.1,
        0.7, 0.2, 0.0,
        0.5, 0.4, 0.2,
        0.3, 0.1, 0.1
      ), ncol = 3, byrow = TRUE)
    )

    block_b = make_block(
      "block_b",
      matrix(c(
        0.8, 0.0, 1.0,
        0.7, 0.2, 0.9,
        0.6, 0.1, 0.7,
        0.4, 0.0, 0.8,
        0.3, 0.2, 0.6,
        0.2, 0.1, 0.5
      ), ncol = 3, byrow = TRUE)
    )

    block_c = make_block(
      "block_c",
      matrix(c(
        0.1, 1.1, 0.8,
        0.2, 1.0, 0.7,
        0.3, 0.9, 0.5,
        0.2, 0.8, 0.4,
        0.0, 0.7, 0.3,
        0.1, 0.6, 0.2
      ), ncol = 3, byrow = TRUE)
    )

    response = as.numeric(base::scale(1.0 * z1 + 0.7 * z2 - 0.4 * z3 + stats::rnorm(n, sd = 0.4)))

    score_class = z1 + 0.5 * z2 - 0.3 * z3 + stats::rnorm(n, sd = 0.3)
    cuts = stats::quantile(score_class, probs = c(1 / 3, 2 / 3), na.rm = TRUE)
    subtype = cut(
      score_class,
      breaks = c(-Inf, cuts[1L], cuts[2L], Inf),
      labels = c("Basal", "Her2", "LumA"),
      include.lowest = TRUE,
      ordered_result = FALSE
    )
    subtype = factor(subtype, levels = c("Basal", "Her2", "LumA"))

    dt = cbind(
      data.table::data.table(site_batch = site_batch),
      block_a,
      block_b,
      block_c,
      data.table::data.table(subtype = subtype, response = response)
    )

    blocks = list(
      block_a = names(block_a),
      block_b = names(block_b),
      block_c = names(block_c)
    )

    list(dt = dt, blocks = blocks)
  })
}




TaskClassifMultiBlock = R6::R6Class(
  "TaskClassifMultiBlock",
  inherit = mlr3::TaskClassif,
  private = list(
    .blocks = NULL
  ),
  public = list(
    initialize = function(id, backend, target, blocks = NULL, positive = NULL, label = NA_character_, extra_args = list()) {
      resolved_blocks = mb_task_resolve_constructor_blocks(
        blocks = blocks,
        extra_args = extra_args,
        context = "TaskClassifMultiBlock$new()"
      )
      extra_args$blocks = resolved_blocks
      super$initialize(
        id = id,
        backend = backend,
        target = target,
        positive = positive,
        label = label,
        extra_args = extra_args
      )
      private$.blocks = mb_drop_target_from_blocks(resolved_blocks, target, context = "TaskClassifMultiBlock$new()")
      mb_assert_columns_present(
        self$col_info$id,
        unique(unlist(private$.blocks, use.names = FALSE)),
        context = "TaskClassifMultiBlock$new()",
        hint = "Ensure that every block column exists in the backend after task construction."
      )
      self$extra_args$blocks = private$.blocks
      self$man = "mlr3mbspls::as_task_multiblock"
    },
    rename = function(old, new) {
      super$rename(old = old, new = new)
      private$.blocks = mb_blocks_rename(private$.blocks, old = old, new = new)
      self$extra_args$blocks = private$.blocks
      invisible(self)
    },
    block_features = function(block = NULL, materialize = FALSE) {
      mb_task_block_features(self, block = block, materialize = materialize)
    },
    block_data = function(rows = NULL, blocks = NULL, as_matrix = FALSE) {
      mb_task_block_data(self, rows = rows, blocks = blocks, as_matrix = as_matrix)
    },
    format = function(...) {
      out = super$format(...)
      block_sizes = vapply(private$.blocks %||% list(), length, integer(1))
      block_desc = if (length(block_sizes)) {
        paste(sprintf("%s[%d]", names(block_sizes), unname(block_sizes)), collapse = ", ")
      } else {
        "<none>"
      }
      c(out, sprintf("Blocks: %d (%s)", length(block_sizes), block_desc))
    },
    overview = function(rows = NULL, blocks = NULL, include_target = TRUE, top_levels = 5L) {
      mb_task_overview(self, rows = rows, blocks = blocks, include_target = include_target, top_levels = top_levels)
    }
  ),
  active = list(
    blocks = function(rhs) {
      if (!missing(rhs)) {
        stop("`blocks` is read-only. Create a new task if the block mapping must change.", call. = FALSE)
      }
      private$.blocks
    },
    block_names = function(rhs) {
      if (!missing(rhs)) {
        stop("`block_names` is read-only.", call. = FALSE)
      }
      names(private$.blocks) %||% character(0)
    }
  )
)


TaskRegrMultiBlock = R6::R6Class(
  "TaskRegrMultiBlock",
  inherit = mlr3::TaskRegr,
  private = list(
    .blocks = NULL
  ),
  public = list(
    initialize = function(id, backend, target, blocks = NULL, label = NA_character_, extra_args = list()) {
      resolved_blocks = mb_task_resolve_constructor_blocks(
        blocks = blocks,
        extra_args = extra_args,
        context = "TaskRegrMultiBlock$new()"
      )
      extra_args$blocks = resolved_blocks
      super$initialize(
        id = id,
        backend = backend,
        target = target,
        label = label,
        extra_args = extra_args
      )
      private$.blocks = mb_drop_target_from_blocks(resolved_blocks, target, context = "TaskRegrMultiBlock$new()")
      mb_assert_columns_present(
        self$col_info$id,
        unique(unlist(private$.blocks, use.names = FALSE)),
        context = "TaskRegrMultiBlock$new()",
        hint = "Ensure that every block column exists in the backend after task construction."
      )
      self$extra_args$blocks = private$.blocks
      self$man = "mlr3mbspls::as_task_multiblock"
    },
    rename = function(old, new) {
      super$rename(old = old, new = new)
      private$.blocks = mb_blocks_rename(private$.blocks, old = old, new = new)
      self$extra_args$blocks = private$.blocks
      invisible(self)
    },
    block_features = function(block = NULL, materialize = FALSE) {
      mb_task_block_features(self, block = block, materialize = materialize)
    },
    block_data = function(rows = NULL, blocks = NULL, as_matrix = FALSE) {
      mb_task_block_data(self, rows = rows, blocks = blocks, as_matrix = as_matrix)
    },
    format = function(...) {
      out = super$format(...)
      block_sizes = vapply(private$.blocks %||% list(), length, integer(1))
      block_desc = if (length(block_sizes)) {
        paste(sprintf("%s[%d]", names(block_sizes), unname(block_sizes)), collapse = ", ")
      } else {
        "<none>"
      }
      c(out, sprintf("Blocks: %d (%s)", length(block_sizes), block_desc))
    },
    overview = function(rows = NULL, blocks = NULL, include_target = TRUE, top_levels = 5L) {
      mb_task_overview(self, rows = rows, blocks = blocks, include_target = include_target, top_levels = top_levels)
    }
  ),
  active = list(
    blocks = function(rhs) {
      if (!missing(rhs)) {
        stop("`blocks` is read-only. Create a new task if the block mapping must change.", call. = FALSE)
      }
      private$.blocks
    },
    block_names = function(rhs) {
      if (!missing(rhs)) {
        stop("`block_names` is read-only.", call. = FALSE)
      }
      names(private$.blocks) %||% character(0)
    }
  )
)


TaskClustMultiBlock = R6::R6Class(
  "TaskClustMultiBlock",
  inherit = mlr3cluster::TaskClust,
  private = list(
    .blocks = NULL
  ),
  public = list(
    initialize = function(id, backend, blocks = NULL, label = NA_character_, extra_args = list()) {
      resolved_blocks = mb_task_resolve_constructor_blocks(
        blocks = blocks,
        extra_args = extra_args,
        context = "TaskClustMultiBlock$new()"
      )
      extra_args$blocks = resolved_blocks
      super$initialize(
        id = id,
        backend = backend,
        label = label
      )
      private$.blocks = resolved_blocks
      mb_assert_columns_present(
        self$col_info$id,
        unique(unlist(private$.blocks, use.names = FALSE)),
        context = "TaskClustMultiBlock$new()",
        hint = "Ensure that every block column exists in the backend after task construction."
      )
      self$extra_args$blocks = private$.blocks
      self$man = "mlr3mbspls::as_task_multiblock"
    },
    rename = function(old, new) {
      super$rename(old = old, new = new)
      private$.blocks = mb_blocks_rename(private$.blocks, old = old, new = new)
      self$extra_args$blocks = private$.blocks
      invisible(self)
    },
    block_features = function(block = NULL, materialize = FALSE) {
      mb_task_block_features(self, block = block, materialize = materialize)
    },
    block_data = function(rows = NULL, blocks = NULL, as_matrix = FALSE) {
      mb_task_block_data(self, rows = rows, blocks = blocks, as_matrix = as_matrix)
    },
    format = function(...) {
      out = super$format(...)
      block_sizes = vapply(private$.blocks %||% list(), length, integer(1))
      block_desc = if (length(block_sizes)) {
        paste(sprintf("%s[%d]", names(block_sizes), unname(block_sizes)), collapse = ", ")
      } else {
        "<none>"
      }
      c(out, sprintf("Blocks: %d (%s)", length(block_sizes), block_desc))
    },
    overview = function(rows = NULL, blocks = NULL, include_target = TRUE, top_levels = 5L) {
      mb_task_overview(self, rows = rows, blocks = blocks, include_target = include_target, top_levels = top_levels)
    }
  ),
  active = list(
    blocks = function(rhs) {
      if (!missing(rhs)) {
        stop("`blocks` is read-only. Create a new task if the block mapping must change.", call. = FALSE)
      }
      private$.blocks
    },
    block_names = function(rhs) {
      if (!missing(rhs)) {
        stop("`block_names` is read-only.", call. = FALSE)
      }
      names(private$.blocks) %||% character(0)
    }
  )
)
