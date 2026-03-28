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
#' The underlying task always stores a named `blocks` mapping and exposes two
#' convenience methods:
#'
#' * `$block_features(block = NULL, materialize = FALSE)`
#' * `$block_data(rows = NULL, blocks = NULL, as_matrix = FALSE)`
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
#'   vector.
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
#'
#' toy_unsup = task_multiblock_synthetic(task_type = "clust")
#' toy_class = task_multiblock_synthetic(task_type = "classif")
#' toy_regr = task_multiblock_synthetic(task_type = "regr")
#'
#' if (requireNamespace("mixOmics", quietly = TRUE)) {
#'   tcga = task_multiblock_breast_tcga(task_type = "classif")
#'   tcga$block_names
#' }
#'
#' if (requireNamespace("multiblock", quietly = TRUE)) {
#'   pot = task_multiblock_potato(task_type = "regr")
#'   pot$block_names
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
    }
    if (is.null(target) && (inherits(x, "TaskClassif") || inherits(x, "TaskRegr"))) {
      target = x$target_names[1L]
    }
    cols = unique(c(x$feature_names, tryCatch(x$target_names, error = function(e) character(0))))
    x = data.table::as.data.table(x$data(rows = x$row_ids, cols = cols))
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
    source_task = source_task
  )

  prep$backend = mb_finalize_target_type(
    dt = prep$backend,
    target = prep$target,
    task_type = resolved_type
  )

  target_col = if (identical(resolved_type, "clust")) NULL else prep$target

  task = switch(
    resolved_type,
    classif = TaskClassifMultiBlock$new(
      id = id,
      backend = prep$backend,
      target = target_col,
      blocks = prep$blocks,
      positive = positive,
      label = label
    ),
    regr = TaskRegrMultiBlock$new(
      id = id,
      backend = prep$backend,
      target = target_col,
      blocks = prep$blocks,
      label = label
    ),
    clust = TaskClustMultiBlock$new(
      id = id,
      backend = prep$backend,
      blocks = prep$blocks,
      label = label
    )
  )

  if (
    is.null(positive) &&
      !is.null(source_task) &&
      inherits(source_task, "TaskClassif") &&
      inherits(task, "TaskClassif")
  ) {
    pos = tryCatch(source_task$positive, error = function(e) NA_character_)
    if (!is.na(pos) && pos %in% task$class_names) {
      task$positive = pos
    }
  }

  task
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

  blocks = mb_pick_named_elements(
    ds,
    groups = list(
      mRNA = c("mRNA", "mrna"),
      miRNA = c("miRNA", "mirna"),
      protein = c("protein")
    ),
    context = "task_multiblock_breast_tcga"
  )

  subtype = ds[[mb_pick_single_name(ds, c("subtype", "Subtype"), context = "task_multiblock_breast_tcga")]]

  TaskMultiBlock(
    x = blocks,
    target = if (identical(task_type, "classif")) subtype else NULL,
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
  if (inherits(sensory, "AsIs")) {
    sensory = unclass(sensory)
  }
  y = NULL
  if (identical(task_type, "regr")) {
    if (is.character(response)) {
      if (!response %in% colnames(sensory)) {
        stop(sprintf("Unknown potato response '%s'.", response), call. = FALSE)
      }
      y = sensory[, response, drop = TRUE]
    } else {
      response = as.integer(response)
      if (response < 1L || response > ncol(sensory)) {
        stop("`response` is out of bounds for potato$Sensory.", call. = FALSE)
      }
      y = sensory[, response, drop = TRUE]
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


mb_pick_single_name = function(x, candidates, context = "multiblock") {
  nms = names(x)
  if (is.null(nms) || !length(nms)) {
    stop(sprintf("%s: input has no named elements.", context), call. = FALSE)
  }

  idx = match(tolower(candidates), tolower(nms))
  idx = idx[!is.na(idx)]
  if (!length(idx)) {
    stop(
      sprintf(
        "%s: none of [%s] found. Available names are: %s",
        context,
        paste(candidates, collapse = ", "),
        paste(nms, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  nms[idx[1L]]
}


mb_pick_named_elements = function(x, groups, context = "multiblock") {
  out = lapply(groups, function(candidates) {
    key = mb_pick_single_name(x, candidates = candidates, context = context)
    x[[key]]
  })
  names(out) = names(groups)
  out
}


mb_is_block_list_input = function(x) {
  if (!is.list(x)) {
    return(FALSE)
  }

  # Standard data.frames with atomic columns are treated as flat backends.
  if (inherits(x, "data.frame") || data.table::is.data.table(x)) {
    return(any(vapply(x, function(col) {
      is.matrix(col) || is.data.frame(col) || data.table::is.data.table(col) ||
        inherits(col, "AsIs") || is.list(col)
    }, logical(1))))
  }

  TRUE
}


mb_block_nrow = function(xb) {
  if (inherits(xb, "AsIs")) {
    xb = unclass(xb)
  }

  if (is.null(xb)) {
    return(NA_integer_)
  }

  nr = if (is.null(dim(xb))) NROW(xb) else nrow(xb)
  if (length(nr) != 1L || !is.finite(nr)) {
    return(NA_integer_)
  }

  as.integer(nr)
}


mb_block_as_data_table = function(xb, bn, sep = "__") {
  if (inherits(xb, "AsIs")) {
    xb = unclass(xb)
  }

  if (is.null(xb)) {
    stop(sprintf("Block '%s' is NULL.", bn), call. = FALSE)
  }

  if (is.matrix(xb)) {
    dt = data.table::as.data.table(xb)
  } else if (data.table::is.data.table(xb)) {
    dt = data.table::copy(xb)
  } else if (is.data.frame(xb)) {
    dt = data.table::as.data.table(xb)
  } else if (is.atomic(xb)) {
    dt = data.table::data.table(value = xb)
  } else {
    stop(sprintf("Block '%s' must be a matrix, data.frame, data.table, or atomic vector.", bn), call. = FALSE)
  }

  if (!ncol(dt)) {
    stop(sprintf("Block '%s' has zero columns.", bn), call. = FALSE)
  }

  if (is.null(names(dt)) || any(!nzchar(names(dt)))) {
    data.table::setnames(dt, paste0("V", seq_len(ncol(dt))))
  }

  data.table::setnames(dt, paste0(bn, sep, make.names(names(dt), unique = TRUE)))
  dt
}


mb_prepare_multiblock_input = function(x, blocks = NULL, target = NULL, target_name = NULL, sep = "__") {
  if (mb_is_block_list_input(x)) {
    if (!length(x)) {
      stop("List input for `TaskMultiBlock()` must contain at least one block.", call. = FALSE)
    }
    if (is.null(names(x)) || any(!nzchar(names(x)))) {
      names(x) = paste0("block_", seq_along(x))
    }

    n_rows = unique(vapply(x, mb_block_nrow, integer(1)))
    if (any(is.na(n_rows))) {
      stop("All blocks must be non-NULL and provide a row count.", call. = FALSE)
    }
    if (length(n_rows) != 1L) {
      stop("All blocks must have the same number of rows.", call. = FALSE)
    }

    pieces = Map(function(xb, bn) {
      mb_block_as_data_table(xb = xb, bn = bn, sep = sep)
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
        if (length(target) != n_rows[1L]) {
          stop("Target vector must have the same number of rows as the blocks.", call. = FALSE)
        }
        target_col = mb_make_target_name(names(dt), target_name)
        dt[[target_col]] = target
      }
    }

    if (!is.null(target_col)) {
      block_map = lapply(block_map, function(cols) setdiff(cols, target_col))
    }

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

  if (!is.null(target_col)) {
    block_map = lapply(block_map, function(cols) setdiff(cols, target_col))
  }

  list(backend = dt, blocks = block_map, target = target_col)
}


mb_infer_task_type = function(task_type, dt, target, source_task = NULL) {
  if (!identical(task_type, "auto")) {
    return(task_type)
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
  if (is.null(target)) {
    return("clust")
  }

  y = dt[[target]]
  if (is.factor(y) || is.character(y) || is.logical(y) || is.ordered(y)) {
    return("classif")
  }
  if (is.numeric(y) || is.integer(y)) {
    return("regr")
  }

  stop("Could not infer `task_type` from the supplied target.", call. = FALSE)
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
    mb_resolve_blocks(dt_task, blocks, numeric_only = FALSE, non_constant = FALSE)
  }

  cols = unique(unlist(blocks_map, use.names = FALSE))
  dt = data.table::as.data.table(task$data(rows = rows, cols = cols))

  out = lapply(blocks_map, function(cols_block) {
    x_block = dt[, cols_block, with = FALSE]
    if (isTRUE(as_matrix)) {
      as.matrix(x_block)
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
  if (!is.na(seed)) {
    set.seed(seed)
  }

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
}


TaskClassifMultiBlock = R6::R6Class(
  "TaskClassifMultiBlock",
  inherit = mlr3::TaskClassif,
  public = list(
    blocks = NULL,
    initialize = function(id, backend, target, blocks, positive = NULL, label = NA_character_) {
      super$initialize(id = id, backend = backend, target = target, positive = positive, label = label)
      self$blocks = mb_normalize_blocks(blocks)
    },
    block_features = function(block = NULL, materialize = FALSE) {
      mb_task_block_features(self, block = block, materialize = materialize)
    },
    block_data = function(rows = NULL, blocks = NULL, as_matrix = FALSE) {
      mb_task_block_data(self, rows = rows, blocks = blocks, as_matrix = as_matrix)
    }
  ),
  active = list(
    block_names = function(rhs) {
      if (!missing(rhs)) {
        stop("`block_names` is read-only.", call. = FALSE)
      }
      names(self$blocks) %||% character(0)
    }
  )
)


TaskRegrMultiBlock = R6::R6Class(
  "TaskRegrMultiBlock",
  inherit = mlr3::TaskRegr,
  public = list(
    blocks = NULL,
    initialize = function(id, backend, target, blocks, label = NA_character_) {
      super$initialize(id = id, backend = backend, target = target, label = label)
      self$blocks = mb_normalize_blocks(blocks)
    },
    block_features = function(block = NULL, materialize = FALSE) {
      mb_task_block_features(self, block = block, materialize = materialize)
    },
    block_data = function(rows = NULL, blocks = NULL, as_matrix = FALSE) {
      mb_task_block_data(self, rows = rows, blocks = blocks, as_matrix = as_matrix)
    }
  ),
  active = list(
    block_names = function(rhs) {
      if (!missing(rhs)) {
        stop("`block_names` is read-only.", call. = FALSE)
      }
      names(self$blocks) %||% character(0)
    }
  )
)


TaskClustMultiBlock = R6::R6Class(
  "TaskClustMultiBlock",
  inherit = mlr3cluster::TaskClust,
  public = list(
    blocks = NULL,
    initialize = function(id, backend, blocks, label = NA_character_) {
      super$initialize(id = id, backend = backend, label = label)
      self$blocks = mb_normalize_blocks(blocks)
    },
    block_features = function(block = NULL, materialize = FALSE) {
      mb_task_block_features(self, block = block, materialize = materialize)
    },
    block_data = function(rows = NULL, blocks = NULL, as_matrix = FALSE) {
      mb_task_block_data(self, rows = rows, blocks = blocks, as_matrix = as_matrix)
    }
  ),
  active = list(
    block_names = function(rhs) {
      if (!missing(rhs)) {
        stop("`block_names` is read-only.", call. = FALSE)
      }
      names(self$blocks) %||% character(0)
    }
  )
)
