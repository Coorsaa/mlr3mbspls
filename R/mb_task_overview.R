#' Summarize a multi-block task
#'
#' @title Block-wise quality-control summary for TaskMultiBlock objects
#'
#' @description
#' `mb_task_overview()` creates a compact, analysis-ready overview of a
#' multi-block task. It is designed as an early quality-control step before
#' fitting MB-sPLS, MB-sPLS-XY, or MB-sPCA pipelines.
#'
#' The returned object contains:
#'
#' * `overview`: task-level dimensions and complete-case rates,
#' * `blocks`: one row per block with feature counts, type counts, missingness,
#'   and constant-feature counts,
#' * `target`: a one-row summary for supervised tasks,
#' * `target_distribution`: class counts for classification tasks,
#' * `issues`: a compact table of actionable warnings/notes.
#'
#' This helper is particularly useful in clinical, social-science, and
#' multi-omics workflows where block imbalance, missingness, and low-information
#' variables should be checked explicitly before resampling or interpretation.
#'
#' For `TaskMultiBlock()` objects, `task$overview()` is the primary interface.
#' `mb_task_overview(task)` is retained as a convenience wrapper for ad hoc code
#' and backward compatibility.
#'
#' @param task A task created with [TaskMultiBlock()] or any [mlr3::Task] that
#'   carries multiblock metadata in `task$blocks` or `task$extra_args$blocks`.
#' @param rows Optional row ids to summarize. Defaults to all task rows.
#' @param blocks Optional block subset. Either a character vector of block names
#'   or a named block mapping.
#' @param include_target Logical; summarize the target if the task is
#'   supervised.
#' @param top_levels Maximum number of classification levels to print in the
#'   target summary. All levels are still returned in `target_distribution`.
#'
#' @return A named list with `data.table` components: `overview`, `blocks`,
#'   `target`, `target_distribution`, and `issues`.
#'
#' @examples
#' task = mlr3::tsk("mbspls_synthetic_classif")
#' qc = task$overview()
#' qc$overview
#' qc$blocks
#' qc$issues
#'
#' @export
mb_task_overview = function(task, rows = NULL, blocks = NULL, include_target = TRUE, top_levels = 5L) {
  checkmate::assert_class(task, "Task", .var.name = "task")
  rows = rows %||% task$row_ids
  checkmate::assert_integerish(rows, any.missing = FALSE, min.len = 1L, .var.name = "rows")
  checkmate::assert_flag(include_target, .var.name = "include_target")
  checkmate::assert_int(top_levels, lower = 1L, .var.name = "top_levels")

  available_blocks = mb_task_blocks(task, context = "mb_task_overview")
  block_map = if (is.null(blocks)) {
    available_blocks
  } else if (is.character(blocks) && is.null(names(blocks))) {
    checkmate::assert_subset(blocks, names(available_blocks), .var.name = "blocks")
    available_blocks[blocks]
  } else {
    mb_normalize_blocks(blocks, .var.name = "blocks")
  }

  block_map = lapply(block_map, function(cols) {
    intersect(mb_expand_block_cols(task$feature_names, cols), task$feature_names)
  })
  block_map = Filter(length, block_map)

  block_cols = unique(unlist(block_map, use.names = FALSE))
  dt = if (length(block_cols)) {
    data.table::as.data.table(task$data(rows = rows, cols = block_cols))
  } else {
    data.table::data.table()
  }

  n_rows = length(rows)
  overall_complete = if (length(block_cols)) sum(stats::complete.cases(dt)) else n_rows
  overview = data.table::data.table(
    task_id = task$id,
    task_type = tryCatch(task$task_type, error = function(e) NA_character_),
    n_rows = n_rows,
    n_blocks = length(block_map),
    n_features = length(block_cols),
    n_complete_rows = overall_complete,
    complete_row_rate = if (n_rows > 0L) overall_complete / n_rows else NA_real_,
    target = NA_character_
  )

  block_rows = lapply(names(block_map), function(block_name) {
    cols_block = block_map[[block_name]]
    xb = if (length(cols_block)) dt[, ..cols_block] else data.table::data.table()

    is_num = vapply(cols_block, function(cl) is.numeric(xb[[cl]]) || is.integer(xb[[cl]]), logical(1))
    is_fac = vapply(cols_block, function(cl) is.factor(xb[[cl]]) || is.ordered(xb[[cl]]), logical(1))
    is_chr = vapply(cols_block, function(cl) is.character(xb[[cl]]), logical(1))
    is_lgl = vapply(cols_block, function(cl) is.logical(xb[[cl]]), logical(1))

    missing_cells = if (length(cols_block)) sum(vapply(cols_block, function(cl) sum(is.na(xb[[cl]])), numeric(1))) else 0
    total_cells = n_rows * length(cols_block)
    complete_rows = if (length(cols_block)) sum(stats::complete.cases(xb)) else n_rows
    n_constant = if (length(cols_block)) {
      sum(vapply(cols_block, function(cl) {
        x = xb[[cl]]
        if (!(is.numeric(x) || is.integer(x))) {
          return(FALSE)
        }
        x = x[!is.na(x)]
        if (!length(x)) {
          return(FALSE)
        }
        length(unique(x)) <= 1L
      }, logical(1)))
    } else {
      0L
    }

    data.table::data.table(
      block = block_name,
      n_features = length(cols_block),
      n_numeric = sum(is_num),
      n_factor = sum(is_fac),
      n_character = sum(is_chr),
      n_logical = sum(is_lgl),
      n_constant_numeric = as.integer(n_constant),
      n_missing_cells = as.integer(missing_cells),
      pct_missing_cells = if (total_cells > 0L) missing_cells / total_cells else 0,
      n_complete_rows = as.integer(complete_rows),
      complete_row_rate = if (n_rows > 0L) complete_rows / n_rows else NA_real_
    )
  })
  blocks_dt = data.table::rbindlist(block_rows, use.names = TRUE, fill = TRUE)

  target_dt = data.table::data.table()
  target_distribution = data.table::data.table()
  issues_rows = list()

  if (nrow(blocks_dt)) {
    for (i in seq_len(nrow(blocks_dt))) {
      row = blocks_dt[i]
      if (row$n_numeric < 1L) {
        issues_rows[[length(issues_rows) + 1L]] = data.table::data.table(
          scope = "block",
          item = row$block,
          severity = "warning",
          issue = "no_numeric_features",
          message = sprintf("Block '%s' has no numeric features after task construction.", row$block)
        )
      }
      if (row$n_constant_numeric > 0L) {
        issues_rows[[length(issues_rows) + 1L]] = data.table::data.table(
          scope = "block",
          item = row$block,
          severity = if (row$n_constant_numeric >= row$n_numeric && row$n_numeric > 0L) "warning" else "note",
          issue = "constant_numeric_features",
          message = sprintf("Block '%s' contains %d constant numeric feature(s).", row$block, row$n_constant_numeric)
        )
      }
      if (row$pct_missing_cells > 0) {
        issues_rows[[length(issues_rows) + 1L]] = data.table::data.table(
          scope = "block",
          item = row$block,
          severity = if (row$pct_missing_cells >= 0.20) "warning" else "note",
          issue = "missing_values",
          message = sprintf("Block '%s' has %.1f%% missing cells.", row$block, 100 * row$pct_missing_cells)
        )
      }
      if (row$complete_row_rate < 0.80) {
        issues_rows[[length(issues_rows) + 1L]] = data.table::data.table(
          scope = "block",
          item = row$block,
          severity = "warning",
          issue = "low_complete_case_rate",
          message = sprintf("Block '%s' has a complete-row rate of %.1f%%.", row$block, 100 * row$complete_row_rate)
        )
      }
    }
  }

  if (isTRUE(include_target)) {
    target_names = tryCatch(task$target_names, error = function(e) character(0))
    if (length(target_names) == 1L) {
      target_name = target_names[[1L]]
      overview[, target := target_name]
      ydt = data.table::as.data.table(task$data(rows = rows, cols = target_name))
      y = ydt[[target_name]]
      target_type = if (inherits(task, "TaskClassif")) {
        "classif"
      } else if (inherits(task, "TaskRegr")) {
        "regr"
      } else {
        class(y)[1L] %||% NA_character_
      }

      if (identical(target_type, "classif")) {
        tab = sort(table(y, useNA = "ifany"), decreasing = TRUE)
        target_distribution = data.table::data.table(
          level = names(tab),
          n = as.integer(tab),
          proportion = as.numeric(tab) / max(1L, sum(tab))
        )
        shown = utils::head(target_distribution, top_levels)
        target_dt = data.table::data.table(
          target = target_name,
          target_type = target_type,
          n_missing = sum(is.na(y)),
          n_levels = length(tab),
          min_class_n = if (length(tab)) min(as.integer(tab)) else NA_integer_,
          max_class_n = if (length(tab)) max(as.integer(tab)) else NA_integer_,
          shown_levels = paste(sprintf("%s=%d", shown$level, shown$n), collapse = "; ")
        )
        if (length(tab) > 0L && min(as.integer(tab)) < 5L) {
          rare = names(tab)[as.integer(tab) == min(as.integer(tab))]
          issues_rows[[length(issues_rows) + 1L]] = data.table::data.table(
            scope = "target",
            item = target_name,
            severity = "warning",
            issue = "small_class_count",
            message = sprintf(
              "Target '%s' contains class(es) with fewer than 5 observations: %s.",
              target_name,
              paste(rare, collapse = ", ")
            )
          )
        }
      } else {
        y_num = as.numeric(y)
        y_obs = y_num[!is.na(y_num)]
        target_dt = data.table::data.table(
          target = target_name,
          target_type = target_type,
          n_missing = sum(is.na(y_num)),
          mean = if (length(y_obs)) mean(y_obs) else NA_real_,
          sd = if (length(y_obs) > 1L) stats::sd(y_obs) else NA_real_,
          min = if (length(y_obs)) min(y_obs) else NA_real_,
          max = if (length(y_obs)) max(y_obs) else NA_real_
        )
        if (length(y_obs) > 1L && isTRUE(all.equal(stats::sd(y_obs), 0))) {
          issues_rows[[length(issues_rows) + 1L]] = data.table::data.table(
            scope = "target",
            item = target_name,
            severity = "warning",
            issue = "constant_target",
            message = sprintf("Target '%s' has zero variance.", target_name)
          )
        }
      }
    }
  }

  issues_dt = if (length(issues_rows)) {
    data.table::rbindlist(issues_rows, use.names = TRUE, fill = TRUE)
  } else {
    data.table::data.table(
      scope = character(),
      item = character(),
      severity = character(),
      issue = character(),
      message = character()
    )
  }

  list(
    overview = overview,
    blocks = blocks_dt,
    target = target_dt,
    target_distribution = target_distribution,
    issues = issues_dt
  )
}
