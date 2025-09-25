#' Feature Suffixer
#'
#' @title PipeOp \code{feature_suffix}: append a suffix to all task feature names
#'
#' @description
#' Renames all feature columns by appending a user-defined \code{suffix}.
#' Targets and other roles are left unchanged. The same mapping is applied
#' during prediction. By default, columns that already end with the suffix
#' are skipped (to prevent double-suffixing).
#'
#' @section Parameters:
#' @param suffix character(1). Suffix to append. Default "_sfx".
#' @param skip_already_suffixed logical(1). If TRUE (default),
#'   skip features that already end with \code{suffix}.
#' @param error_on_collision logical(1). If TRUE (default), error if
#'   any new names would collide with non-feature columns or create duplicates.
#'
#' @section Construction:
#' `PipeOpFeatureSuffix$new(id = "feature_suffix", param_vals = list())`
#'
#' @section Methods:
#' * `$new(id, param_vals)` : Initialize the PipeOpFeatureSuffix.
#'
#' @param id character(1). Identifier of the resulting object.
#' @param param_vals named list. List of hyperparameter settings, overwriting the hyperparameter settings that would otherwise be set during construction.
#'
#' @importFrom R6 R6Class
#' @import lgr
#' @importFrom checkmate assert_string assert_flag
#' @importFrom paradox ps p_uty p_lgl
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @export
PipeOpFeatureSuffix = R6::R6Class(
  "PipeOpFeatureSuffix",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @description Initialize the PipeOpFeatureSuffix.
    #' @param id character(1). Identifier of the resulting object.
    #' @param param_vals named list. List of hyperparameter settings.
    initialize = function(id = "feature_suffix", param_vals = list()) {
      ps = paradox::ps(
        suffix                 = p_uty(tags = c("train", "predict"), default = "_sfx"),
        skip_already_suffixed  = p_lgl(default = TRUE, tags = c("train", "predict")),
        error_on_collision     = p_lgl(default = TRUE, tags = c("train", "predict"))
      )
      super$initialize(id = id, param_set = ps, param_vals = param_vals)
    }
  ),

  private = list(

    .apply_suffix = function(task, stage = c("train", "predict")) {
      stage = match.arg(stage)

      pv = utils::modifyList(
        paradox::default_values(self$param_set),
        self$param_set$get_values(tags = stage),
        keep.null = TRUE
      )
      checkmate::assert_string(pv$suffix, min.chars = 0L)
      checkmate::assert_flag(pv$skip_already_suffixed)
      checkmate::assert_flag(pv$error_on_collision)

      # Prediction: reuse the exact mapping from training if available
      if (stage == "predict" && length(self$state$rename_map)) {
        map = self$state$rename_map
        olds = intersect(names(map), task$feature_names)
        if (length(olds)) {
          news = unname(map[olds])
          task$rename(old = olds, new = news) # explicit args: old -> new
        } else {
          # If everything is already suffixed, pass through silently
          if (all(unname(map) %in% task$feature_names)) {
            lgr$debug("[%s] Predict: features already carry suffix; pass-through.", self$id)
          } else {
            lgr$warn("[%s] Predict: expected features not found; pass-through.", self$id)
          }
        }
        return(task)
      }

      # Training: build mapping from *current* feature names
      feats = task$feature_names
      if (!length(feats) || identical(pv$suffix, "")) {
        self$state$rename_map = setNames(character(0), character(0))
        return(task)
      }

      # Optionally skip names that already end with the suffix
      todo = if (isTRUE(pv$skip_already_suffixed)) feats[!endsWith(feats, pv$suffix)] else feats
      if (!length(todo)) {
        lgr$info("[%s] Train: all features already have suffix '%s'; nothing to do.", self$id, pv$suffix)
        self$state$rename_map = setNames(character(0), character(0))
        return(task)
      }

      new_names = paste0(todo, pv$suffix)

      # Defensive checks to avoid backend rename failures
      if (isTRUE(pv$error_on_collision)) {
        # All columns in the backend (features + targets + others)
        all_cols = task$backend$colnames
        # Columns that will remain unchanged after rename
        unchanged = setdiff(all_cols, todo)

        # 1) New names must be unique together with unchanged names
        if (anyDuplicated(c(unchanged, new_names))) {
          dup = unique(new_names[duplicated(c(unchanged, new_names))])
          stop(sprintf("[%s] Duplicate column name(s) after renaming: %s", self$id, paste(dup, collapse = ", ")))
        }
        # 2) New names must not collide with non-feature columns
        coll = intersect(new_names, setdiff(all_cols, feats))
        if (length(coll)) {
          stop(sprintf("[%s] Renaming would collide with non-feature columns: %s",
            self$id, paste(coll, collapse = ", ")))
        }
      }

      # Perform rename with explicit argument names (old -> new)
      task$rename(old = todo, new = new_names)

      # Remember exact mapping for predict()
      self$state$rename_map = stats::setNames(new_names, todo)
      task
    },

    .train_task = function(task) private$.apply_suffix(task, "train"),
    .predict_task = function(task) private$.apply_suffix(task, "predict"),

    .additional_phash_input = function() {
      list(
        suffix = self$param_set$values$suffix,
        skip_already_suffixed = self$param_set$values$skip_already_suffixed
      )
    }
  )
)
