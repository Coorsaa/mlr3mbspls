#' Target-Label Filter
#'
#' @title PipeOp \code{target_label_filter}: filter observations by target label(s)
#'
#' @description
#' \strong{PipeOpTargetLabelFilter} filters the input \code{Task} by its target
#' column and forwards the filtered \code{Task}. It is a dedicated, composable
#' row filter that replaces ad-hoc filtering inside other PipeOps.
#'
#' By default, filtering is applied during training. During prediction, if the
#' target column is not present (typical for newdata), the operator becomes a
#' no-op and passes the task through unchanged.
#'
#' @section Parameters:
#' @param labels \code{vector} or \code{NULL}.
#'   Target label(s) used for filtering.
#'   If \code{invert = FALSE} (default), rows with \code{target \%in\% labels} are kept.
#'   If \code{invert = TRUE}, rows with \code{target \%in\% labels} are dropped.
#'   If \code{NULL}, the task is passed through unchanged.
#' @param target \code{character(1)} or \code{NULL}. Name of the target column.
#'   Defaults to the task's first target via \code{task$target_names[1]}.
#' @param invert \code{logical(1)}. If \code{TRUE}, invert the selection.
#' @param drop_unused_levels \code{logical(1)}. If \code{TRUE} (default),
#'   drop unused factor levels on \emph{non-target} factor columns after filtering.
#'   The target's level set is controlled explicitly (see Details).
#' @param drop_stratum \code{logical(1)}. If \code{TRUE} (default \code{FALSE}),
#'   remove the \code{"stratum"} role from columns that are neither features nor targets.
#'
#' @details
#' After filtering, if the target is a factor:
#' \itemize{
#'   \item For \code{invert = FALSE} the target's level set is the union of
#'         \code{labels} with their original order preserved (unobserved labels are allowed).
#'   \item For \code{invert = TRUE} the target's level set is \code{setdiff(original_levels, labels)}.
#'   \item If this level set has fewer than two levels, it is padded with additional
#'         level(s) from the original levels (as needed) so that \code{length(levels(target)) >= 2}.
#' }
#' Some learners still require \emph{two observed classes} in the training data;
#' this PipeOp only guarantees the metadata (level set), not the label balance.
#'
#' @return A \code{PipeOpTargetLabelFilter}.
#'
#' @family PipeOps
#' @keywords internal
#' @importFrom R6 R6Class
#' @import data.table lgr
#' @importFrom checkmate assert_flag
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
      ps = paradox::ps(
        labels             = p_uty(tags = c("train", "predict"), default = NULL),
        target             = p_uty(tags = c("train", "predict"), default = NULL),
        invert             = p_lgl(default = FALSE, tags = c("train", "predict")),
        drop_unused_levels = p_lgl(default = TRUE, tags = c("train", "predict")),
        drop_stratum       = p_lgl(default = FALSE, tags = c("train", "predict"))
      )
      super$initialize(id = id, param_set = ps, param_vals = param_vals)
    }
  ),

  private = list(

    .apply_filter = function(task, stage = c("train", "predict")) {
      stage = match.arg(stage)

      # Resolve params for this stage
      pv = utils::modifyList(
        paradox::default_values(self$param_set),
        self$param_set$get_values(tags = stage),
        keep.null = TRUE
      )

      # Optionally drop the 'stratum' role from columns that are neither features nor targets
      if (isTRUE(pv$drop_stratum)) {
        strata_cols = task$col_roles$stratum
        if (length(strata_cols)) {
          keep_cols = unique(c(task$feature_names, task$target_names))
          drop_cols = setdiff(strata_cols, keep_cols)
          if (length(drop_cols)) {
            # Remove ALL roles from these columns (they were not feature/target anyway)
            task$set_col_roles(drop_cols, roles = character(0))
            lgr$info("[%s] Dropped stratum role from %d column(s): %s",
              self$id, length(drop_cols), paste(drop_cols, collapse = ", "))
          }
        }
      }

      # Determine target column
      trg = pv$target
      if (is.null(trg) || !is.character(trg) || length(trg) != 1L) {
        trg = if (length(task$target_names)) task$target_names[1L] else NULL
      }
      if (is.null(trg) || !(trg %in% task$target_names)) {
        if (stage == "train") {
          lgr$warn("[%s] No valid target column found; passing task through unchanged.", self$id)
        }
        return(task)
      }

      # If no labels were provided -> pass-through
      if (is.null(pv$labels)) {
        return(task)
      }

      # Compute keep mask from current data
      dt = task$data(cols = trg)
      y = dt[[trg]]
      keep = y %in% pv$labels
      if (isTRUE(pv$invert)) keep <- !keep

      if (!any(keep)) {
        stop(sprintf("[%s] Filtering by target='%s' and labels=%s produced 0 rows.",
          self$id, trg, paste(utils::head(as.character(pv$labels), 3L), collapse = ", ")))
      }

      # Apply row filter
      keep_ids = task$row_ids[keep]
      n_drop = length(task$row_ids) - length(keep_ids)
      if (n_drop > 0L) {
        lgr$info("[%s] Removed %d / %d rows by target filter.",
          self$id, n_drop, length(task$row_ids))
      }
      task$filter(keep_ids)

      # ---- Factor-level handling on the TARGET ---------------------------------
      if (is.factor(y)) {
        orig_levels = levels(y)
        labels_chr = as.character(pv$labels)

        # Desired level set after filtering
        desired = if (!isTRUE(pv$invert)) {
          # Keep all labels (unobserved allowed), in original order where possible
          unique(c(orig_levels[orig_levels %in% labels_chr], setdiff(labels_chr, orig_levels)))
        } else {
          # Keep all "others"
          setdiff(orig_levels, labels_chr)
        }

        # Ensure at least two levels in the LEVEL SET (not necessarily observed)
        if (length(desired) < 2L) {
          extra = setdiff(orig_levels, desired)
          if (length(extra)) {
            need = 2L - length(desired)
            desired = c(desired, head(extra, need))
          }
          if (length(desired) < 2L) {
            # Still < 2 (e.g., original task itself had < 2 levels)
            lgr$warn("[%s] Target '%s' has < 2 distinct levels available; proceeding with %d level(s).",
              self$id, trg, length(desired))
          }
        }

        # Apply the level set to the target only
        levlist = list()
        levlist[[trg]] = desired
        task$set_levels(levlist)

        # Optionally drop unused levels on *other* factor columns
        if (isTRUE(pv$drop_unused_levels)) {
          ci = task$col_info
          other_fct = setdiff(ci$id[ci$type %in% c("factor", "ordered")], trg)
          if (length(other_fct)) task$droplevels(cols = other_fct)
        }

        # Keep 'positive' valid if this is a classification task
        if (inherits(task, "TaskClassif")) {
          pos = tryCatch(task$positive, error = function(e) NA_character_)
          if (!is.na(pos) && !(pos %in% desired) && length(desired)) {
            task$positive = desired[1L]
          }
        }

      } else {
        # Non-factor target: optionally drop unused levels in the rest
        if (isTRUE(pv$drop_unused_levels)) task$droplevels()
      }

      task
    },

    .train_task = function(task) private$.apply_filter(task, stage = "train"),
    .predict_task = function(task) private$.apply_filter(task, stage = "predict"),

    .additional_phash_input = function() {
      vals = self$param_set$values
      list(
        target = vals$target,
        labels = vals$labels,
        invert = vals$invert,
        drop_unused_levels = vals$drop_unused_levels,
        drop_stratum = vals$drop_stratum
      )
    }
  )
)
