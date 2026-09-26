#' Replace transformed feature columns of a task in place
#'
#' Removes `drop` from the feature role and replaces the `changed` feature
#' columns by the corresponding columns of `data` via `Task$select()` and
#' `Task$cbind()`, the in-place pattern used by
#' [mlr3pipelines::PipeOpTaskPreproc]. Column information (types, levels,
#' labels), the backend primary key and all other column roles therefore stay
#' consistent with the backend. The original feature order is preserved.
#'
#' @param task ([mlr3::Task]) Task to modify in place.
#' @param data (`data.table`) Transformed data whose rows follow
#'   `task$row_ids`.
#' @param changed (`character()`) Feature columns to replace from `data`.
#' @param drop (`character()`) Feature columns to remove from the feature role;
#'   takes precedence over `changed`. Other roles of these columns are kept.
#' @return The modified `task`, invisibly.
#' @noRd
mb_task_replace_features = function(task, data, changed, drop = character(0)) {
  features = task$feature_names
  drop = intersect(drop, features)
  changed = setdiff(intersect(changed, features), drop)
  new_features = setdiff(features, drop)
  task$select(setdiff(new_features, changed))
  if (length(changed)) {
    task$cbind(data[, changed, with = FALSE])
  }
  task$col_roles$feature = new_features
  invisible(task)
}
