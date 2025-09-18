#' Target-Label Filter
#'
#' @title PipeOp \code{target_label_filter}: filter observations by target label(s)
#'
#' @description
#' \strong{PipeOpTargetLabelFilter} filters the input \code{Task} by its target
#' column and forwards the filtered \code{Task}. It is intended as a dedicated,
#' composable row filter that replaces ad-hoc row filtering inside other PipeOps.
#'
#' By default, filtering is applied during training. During prediction, if the
#' target column is not present (typical for newdata), the operator becomes a
#' no-op and passes the task through unchanged.
#'
#' @section Parameters:
#' @param labels \code{vector} or \code{NULL}.
#'  Target label(s) used for filtering.
#'  If \code{invert = FALSE} (default), rows with \code{target %in% labels} are kept.
#'  If \code{invert = TRUE}, rows with \code{target %in% labels} are dropped.
#'  If \code{NULL}, the task is passed through unchanged.
#' @param target \code{character(1)} or \code{NULL}. Name of the target column.
#'   Defaults to the task's first target via \code{task$target_names[1]}.
#' @param invert \code{logical(1)}. If \code{TRUE}, invert the selection.
#' @param drop_unused_levels \code{logical(1)}. If \code{TRUE} (default),
#'   unused factor levels of the target are dropped after filtering.
#'
#' @return A \code{PipeOpTargetLabelFilter} that filters rows of the input task.
#'
#' @family PipeOps
#' @keywords internal
#' @importFrom R6 R6Class
#' @import data.table lgr
#' @importFrom checkmate assert_string assert_flag
#' @importFrom paradox ps p_lgl p_uty
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @export
PipeOpTargetLabelFilter = R6::R6Class(
  "PipeOpTargetLabelFilter",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @description Create a new PipeOpTargetLabelFilter object.
    #' @param id character(1). Identifier (default: "target_label_filter").
    #' @param param_vals list. Initial ParamSet values.
    initialize = function(id = "target_label_filter", param_vals = list()) {
      ps <- paradox::ps(
        labels             = p_uty(tags = c("train", "predict"), default = NULL),
        target             = p_uty(tags = c("train", "predict"), default = NULL),
        invert             = p_lgl(default = FALSE, tags = c("train", "predict")),
        drop_unused_levels = p_lgl(default = TRUE,  tags = c("train", "predict"))
      )
      super$initialize(id = id, param_set = ps, param_vals = param_vals)
    }
  ),

  private = list(

    .apply_filter = function(task, stage = c("train", "predict")) {
      stage <- match.arg(stage)

      # Merge defaults with current stage values
      pv <- utils::modifyList(
        paradox::default_values(self$param_set),
        self$param_set$get_values(tags = stage),
        keep.null = TRUE
      )

      # Determine target column
      trg <- pv$target
      if (is.null(trg) || !is.character(trg) || length(trg) != 1L) {
        trg <- if (length(task$target_names)) task$target_names[1L] else NULL
      }
      if (is.null(trg) || !(trg %in% task$target_names)) {
        if (stage == "train") {
          lgr$warn("[%s] No valid target column found; passing task through unchanged.", self$id)
        }
        return(task)
      }

      # If the target column is not present in the backend (e.g. predict newdata), pass-through
      dt <- task$data(cols = trg)
      if (!trg %in% names(dt)) {
        if (stage == "train") {
          lgr$warn("[%s] Target column '%s' not present in task$data(); passing through.", self$id, trg)
        }
        return(task)
      }

      # No labels provided → pass-through
      if (is.null(pv$labels)) return(task)

      y <- dt[[trg]]
      keep <- y %in% pv$labels
      if (isTRUE(pv$invert)) keep <- !keep

      if (!any(keep)) {
        stop(sprintf("[%s] Filtering by target='%s' and labels=%s produced 0 rows.",
          self$id, trg, paste(utils::head(as.character(pv$labels), 3L), collapse = ", ")))
      }

      keep_ids <- task$row_ids[keep]
      n_drop   <- length(task$row_ids) - length(keep_ids)
      if (n_drop > 0L) {
        lgr$info("[%s] Removed %d / %d rows by target filter.", self$id, n_drop, length(task$row_ids))
      }

      task$filter(keep_ids)
      if (isTRUE(pv$drop_unused_levels)) task$droplevels()
      task
    },

    .train_task   = function(task) private$.apply_filter(task, stage = "train"),
    .predict_task = function(task) private$.apply_filter(task, stage = "predict"),

    .additional_phash_input = function() {
      vals <- self$param_set$values
      list(target = vals$target, labels = vals$labels, invert = vals$invert, drop_unused_levels = vals$drop_unused_levels)
    }
  )
)
