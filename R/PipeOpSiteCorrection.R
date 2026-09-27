#' @title Site-/Batch-Effect Correction PipeOp (block-specific, lean)
#' @name PipeOpSiteCorrection
#' @rdname PipeOpSiteCorrection
#' @format R6 class inheriting from [mlr3pipelines::PipeOpTaskPreproc]
#'
#' @description
#' `PipeOpSiteCorrection` removes unwanted **site / batch effects** from
#' **multi-block data** inside an *mlr3* pipeline. Behavior is controlled by
#' two named lists (by **block**):
#'
#' - `site_correction`: block -> **specification of site/batch and optional covariates**.
#'   - For `"partial_corr"`: a **character vector of columns** (categorical site and/or numeric covariates).
#'   - For `"dir"`: a **single categorical column** (protected attribute).
#'   - For `"combat"` (**new structured format**): a **list** with fixed elements
#'     `list(site = <character(1)>, covariates = <character()>)`.
#'     The `site` column is mapped to ComBat's `batch`; the `covariates` are
#'     encoded into a model matrix and passed as `mod`.
#'     *(Backward compatible: if a single string is provided, it is treated as `site`.)*
#' - `method`: `block -> "partial_corr" | "combat" | "dir"`. Missing blocks
#'   default to `"partial_corr"`.
#'
#' Blocks **absent** from `site_correction` are **left unchanged**.
#'
#' Columns that are read at prediction time (all `"partial_corr"` and `"dir"`
#' columns and the ComBat `site`) must not be target columns: the correction of
#' assessment rows would otherwise depend on their labels, and unlabeled new
#' data could not be corrected. Only ComBat `covariates` may reference a target,
#' because they are used at training time only (see below).
#'
#' @details
#' All correction parameters are estimated from the training rows only and are
#' applied unchanged at prediction; prediction data are never used to
#' re-estimate site effects, batch parameters or repair maps.
#'
#' **Partial correlation (`"partial_corr"`)**
#' - If the site spec is a **single categorical** column, we build a dummy-coded
#'   design with an intercept and keep a stable column layout across train/predict.
#'   At predict-time, **unseen or missing site labels** receive no site-specific
#'   correction under `unknown_site = "other"` (default), or are mapped to the
#'   baseline (first training level) under `unknown_site = "baseline"`. With
#'   `"other"`, such rows share the output location of the corrected rows: they
#'   are returned unchanged when `zero_center = FALSE` and minus the training
#'   grand means when `zero_center = TRUE`.
#' - If the site spec is **multiple columns** (e.g., site plus covariates such
#'   as age or PRS PCs) or numeric, we construct the design as `cbind(1, Z)` via
#'   `model.matrix(~ .)`. Categorical columns of such designs keep their
#'   training levels: prediction rows with unseen levels (including an unseen
#'   site) are rejected and `unknown_site` does not apply. Use a single
#'   categorical site column for unseen-site handling.
#' - We solve a (ridge-stabilized) normal equation for the site effects and
#'   subtract them. The training grand means are re-added unless
#'   `zero_center = TRUE`.
#' - With unpenalized fitting (`regularization = 0`), a site with a single
#'   fitting row is fitted exactly: that row's corrected values carry no
#'   within-site information (with the default settings they equal the training
#'   grand means) and the site offset applied to new rows of that site rests on
#'   one observation. A warning is emitted in this case. A positive
#'   `regularization` shrinks the site coefficients and avoids the exact fit;
#'   alternatively, merge or exclude very small sites.
#' - `subgroup` restricts the rows used to **fit** the site effects (e.g.
#'   controls only); the correction is applied to all rows. It is only used by
#'   `"partial_corr"` blocks and can be given as task row ids (numeric), as the
#'   name of a logical column, or as `list(column = <character(1)>, values = <vector>)`
#'   selecting rows whose `column` value is in `values`. Row ids and columns
#'   travel with the data, so the selection is resampling-safe: ids of rows
#'   outside the current training set are ignored, while ids that do not exist
#'   in the task's backend (e.g. row positions of a task whose row ids are not
#'   `1..n`) are an error. A logical vector is interpreted positionally (one
#'   entry per row of the training task) and is therefore only suitable for
#'   direct use outside resampling.
#'
#' **ComBat (`"combat"`, via \pkg{neuroCombat})**
#' - \pkg{neuroCombat} is distributed on GitHub only; install it with
#'   `remotes::install_github("Jfortin1/neuroCombat_Rpackage@fbec46a61bc92bedb450b0e44addae4ce6afa934")`,
#'   the revision used in continuous integration.
#' - Trains using `neuroCombat(dat = t(X), batch = site, mod = MM, ...)`, where
#'   `MM = model.matrix(~ ., data = covariates)`; character and logical
#'   covariates are factorized. Only site and covariate levels observed in the
#'   training rows are used, so training folds that lack a site (e.g.
#'   leave-site-out resampling) are supported; at least two observed batches
#'   are required. ComBat accepts **one** batch vector, but **many covariates**.
#'   We store the returned `estimates`, the valid batch levels, the `site_var`,
#'   and the list of `covariates`.
#' - At predict, we apply `neuroCombatFromTraining(dat, batch, estimates)`.
#'   The upstream function **does not support** supplying `mod` for new data;
#'   if estimates were trained with `mod` (i.e., `covariates` is non-empty),
#'   the batch-effect removal at predict-time uses the training covariate-effect
#'   estimates only and does NOT re-apply the covariate model to new observations.
#'   **A `warning()` is emitted at predict-time** whenever `covariates` is non-empty
#'   to alert users of this limitation. If predict-time covariate correction matters,
#'   consider using `"partial_corr"` instead. Covariates, including a target
#'   column, are therefore only used to estimate the batch parameters from the
#'   training rows; prediction never reads them and applies the same batch
#'   adjustment to every row of a batch. In unbalanced designs, preserving a
#'   group variable during batch correction can exaggerate downstream group
#'   differences (Nygaard et al., 2016).
#'   Unseen batches can be handled with
#'   `combat_unknown = "noop"` (skip) or `"baseline"` (map to `ref_batch`).
#'
#' **DIR (`"dir"`, geometric disparate impact repair)**
#' - Implements the geometric repair of Feldman et al. (2015) per feature:
#'   \deqn{x \mapsto (1 - \lambda) x + \lambda Q_{med}(F_g(x)),}{x -> (1 - lambda) x + lambda * Q_med(F_g(x)),}
#'   where \eqn{F_g} is the empirical distribution function of the training
#'   values of protected group \eqn{g} (linear interpolation between order
#'   statistics, mid-ranks for ties; the inverse of type-7 quantiles) and
#'   \eqn{Q_{med}}{Q_med} is the pointwise median of the groups' type-7 quantile
#'   functions. The repair is rank-preserving within groups, continuous in
#'   `lambda`, the identity at `lambda = 0`, and it does not quantize the data.
#' - The fitted map of each feature and group is the piecewise-linear
#'   interpolant of this transform through the group's training values; the
#'   training output is the map evaluated at the training values. Prediction
#'   evaluates the same maps by linear interpolation, carrying the repair shift
#'   at the nearest boundary forward beyond the training range (out-of-range
#'   values are shifted, not clipped). The maps are exact at the training
#'   values; between them they approximate the exact transform, and the
#'   approximation is coarsest for groups with few training rows next to much
#'   larger groups, whose quantile functions vary between a small group's
#'   knots. Distributions are never re-estimated from prediction data. For
#'   `lambda > 0`, every group needs at least two distinct training values per
#'   feature. Missing or unseen protected groups are rejected at prediction.
#'
#' The operator updates the task in place (`Task$select()` / `Task$cbind()`, as
#' in [mlr3pipelines::PipeOpTaskPreproc]): corrected features are returned as
#' numeric columns, and column information, backend keys and all column roles
#' stay consistent. By default, all site/covariate columns referenced in
#' `site_correction` are removed from the features unless `keep_site_col = TRUE`
#' (other roles of these columns are kept).
#'
#' @section Construction:
#' \preformatted{
#' PipeOpSiteCorrection$new(
#'   id         = "sitecorr",
#'   param_vals = list()
#' )
#' }
#'
#' @param id `character(1)` Identifier for the new object. Default: `"sitecorr"`.
#' @param param_vals `list()` Named list of hyper-parameter values overriding defaults.
#'
#' @section Tunable hyper-parameters (`self$param_set`):
#' \describe{
#'   \item{**Core**}{
#'     \itemize{
#'       \item `blocks` (`list()`): Named list of **feature vectors** per block
#'         (non-numeric features are auto-dropped). If `NULL`, all numeric
#'         features form one block `".all"`.
#'       \item `site_correction` (`list()`): Named list **by block**. For `"combat"`
#'         use `list(site=<char1>, covariates=<char_vec>)`; for other methods,
#'         character vectors as described above. Missing block => no correction.
#'       \item `method` (`list()`): Named list `block -> "partial_corr"|"combat"|"dir"`.
#'         Missing block => `"partial_corr"`.
#'       \item `keep_site_col` (`logical(1)`): Keep all site/covariate columns referenced by
#'         `site_correction`? Default `FALSE`.
#'     }
#'   }
#'   \item{**Partial-correlation**}{
#'     \itemize{
#'       \item `unknown_site` (`"other"|"baseline"`): Predict-time strategy for an
#'         unseen or missing site when the `"partial_corr"` spec is a **single
#'         categorical** column. Ignored for multi-column or numeric designs,
#'         which reject unseen levels. Default `"other"`.
#'       \item `zero_center` (`logical(1)`): If `TRUE`, return the site-corrected
#'         values centred at zero. If `FALSE` (default), add the training grand
#'         means (column means of all training rows, stored in
#'         `$state$per_block[[b]]$means`) back, so the corrected features stay on
#'         the original scale.
#'       \item `revertflag` (`logical(1)`): Add instead of subtract the site effect.
#'       \item `regularization` (`numeric(1)`): Ridge penalty to stabilize the site
#'         regression (the intercept is not penalized). A value of `0` requests
#'         unpenalized least squares.
#'       \item `subgroup` (`integer()`/`character(1)`/`list()`/`logical()`):
#'         Optional subset of training rows used to fit the site effects: task
#'         row ids, the name of a logical column, `list(column = , values = )`,
#'         or a positional logical vector (not resampling-safe). Only used by
#'         `"partial_corr"` blocks. Default `NULL` (all training rows).
#'     }
#'   }
#'   \item{**ComBat (neuroCombat)**}{
#'     \itemize{
#'       \item `eb` (`logical(1)`): Empirical Bayes shrinkage. Default `TRUE`.
#'       \item `mean_only` (`logical(1)`): Adjust means only. Default `FALSE`.
#'       \item `ref_batch` (`character(1)` or `NULL`): Optional reference batch;
#'         must be observed in the training rows.
#'       \item `combat_unknown` (`"noop"|"baseline"`): Predict-time policy for
#'         unseen batches. Default `"noop"`.
#'     }
#'   }
#'   \item{**DIR**}{
#'     \itemize{
#'       \item `lambda` (`numeric(1)` in \[0,1\]): Repair strength of the
#'         geometric repair; `0` is the identity and `1` maps every group onto
#'         the median distribution. Default `0.5`.
#'     }
#'   }
#'   \item{**Misc**}{
#'     \itemize{
#'       \item `verbose` (`logical(1)`): Emit log messages via \pkg{lgr}.
#'     }
#'   }
#' }
#'
#' @section State (after `$train()`):
#' A named list with:
#' \itemize{
#'   \item `blocks`: Named list of **effective** block feature vectors used for training
#'         (with any referenced site/covariate columns removed).
#'   \item `per_block`: Named list with one entry per corrected block:
#'     \itemize{
#'       \item `method`: `"partial_corr"|"combat"|"dir"`.
#'       \item `site_cols`: Character vector of all referenced site/covariate columns for that block.
#'       \item `design_kind`, `design_cols`, `beta`, `means`, `zero_center`, `revert` (partial_corr).
#'       \item `site_var` (`character(1)`), `covariates` (`character()`), `site_lvls`,
#'             `estimates`, `ref_batch` (combat).
#'       \item `lambda`, `site_lvls`, `repair_maps` (dir; per feature and group,
#'             the training values `x` and the repair shifts `shift`).
#'     }
#'   \item `unknown_site`, `keep_site_col`.
#' }
#'
#' @section Methods:
#' \describe{
#'   \item{`$train(task)`}{Fit correction parameters from an [mlr3::Task].}
#'   \item{`$predict(task)`}{Apply the learnt correction and return a harmonized `Task`.}
#' }
#'
#' @return An [mlr3::Task] with harmonized features.
#' Site/covariate columns referenced by `site_correction` are dropped unless `keep_site_col = TRUE`.
#'
#' @references
#' Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., and
#' Venkatasubramanian, S. (2015). Certifying and removing disparate impact.
#' *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge
#' Discovery and Data Mining*, 259-268. \doi{10.1145/2783258.2783311}
#'
#' Nygaard, V., Rodland, E. A., and Hovig, E. (2016). Methods that remove batch
#' effects while retaining group differences may lead to exaggerated
#' confidence in downstream analyses. *Biostatistics*, 17(1), 29-39.
#' \doi{10.1093/biostatistics/kxv027}
#'
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3pipelines)
#' library(data.table)
#' task = tsk("pima")
#' if (requireNamespace("neuroCombat", quietly = TRUE)) {
#'   numeric_features = task$feature_names
#'   backend = task$data()
#'   backend = backend[stats::complete.cases(backend)]
#'   backend[, site := sample(LETTERS[1:3], .N, TRUE)]
#'   backend[, age_adjustment := rnorm(.N)]
#'   backend[, sex_adjustment := sample(c("F", "M"), .N, TRUE)]
#'   task = TaskClassif$new(
#'     "pima_site",
#'     backend,
#'     target = "diabetes",
#'     positive = "pos"
#'   )
#'
#'   po_site = PipeOpSiteCorrection$new(param_vals = list(
#'     blocks = list(num = numeric_features),
#'     site_correction = list(
#'       num = list(
#'         site = "site",
#'         covariates = c("age_adjustment", "sex_adjustment")
#'       )
#'     ),
#'     method = list(num = "combat"),
#'     keep_site_col = FALSE,
#'     combat_unknown = "noop"
#'   ))
#'   as_graph(po_site)$train(task)
#' }
#' }
#'
#' @family PipeOps
#' @seealso
#'   [mlr3pipelines::PipeOpTaskPreproc],
#'   [neuroCombat::neuroCombat] / [neuroCombat::neuroCombatFromTraining].
#'
#' @importFrom R6 R6Class
#' @importFrom lgr lgr
#' @importFrom data.table := as.data.table
#' @import paradox
#' @importFrom mlr3 as_data_backend
#' @export
PipeOpSiteCorrection = R6::R6Class(
  "PipeOpSiteCorrection",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @description
    #' Create a new `PipeOpSiteCorrection` instance.
    initialize = function(id = "sitecorr", param_vals = list()) {
      ps = paradox::ps(
        # core
        blocks          = p_uty(tags = "train", default = NULL),
        site_correction = p_uty(tags = "train", default = NULL),
        method          = p_uty(tags = "train", default = NULL),
        keep_site_col   = p_lgl(default = FALSE, tags = c("train", "predict")),

        # partial-corr
        unknown_site    = p_fct(c("other", "baseline"), default = "other", tags = c("train", "predict")),
        zero_center     = p_lgl(default = FALSE, tags = c("train", "predict")),
        revertflag      = p_lgl(default = FALSE, tags = c("train", "predict")),
        subgroup        = p_uty(default = NULL, tags = "train"),
        regularization  = p_dbl(lower = 0, default = 0, tags = "train"),

        # ComBat (neuroCombat only)
        eb              = p_lgl(default = TRUE, tags = "train"),
        mean_only       = p_lgl(default = FALSE, tags = "train"),
        ref_batch       = p_uty(default = NULL, tags = "train"),
        combat_unknown  = p_fct(c("noop", "baseline"), default = "noop", tags = c("train", "predict")),

        # DIR
        lambda          = p_dbl(lower = 0, upper = 1, default = 0.5, tags = c("train", "predict")),

        # misc
        verbose         = p_lgl(default = FALSE, tags = c("train", "predict"))
      )

      super$initialize(
        id            = id,
        param_set     = ps,
        param_vals    = param_vals,
        feature_types = c("numeric", "integer", "factor", "character")
      )
    }
  ),

  private = list(

    # ---- helpers ------------------------------------------------------------

    .validate_blocks = function(task, blocks) {
      dt = task$data(cols = task$feature_names)
      if (is.null(blocks)) {
        blocks = mb_task_blocks(task, context = "PipeOpSiteCorrection", allow_null = TRUE)
      }
      if (is.null(blocks)) {
        num = task$feature_names[
          vapply(task$feature_names, function(x) is.numeric(dt[[x]]), logical(1))]
        return(list(.all = num))
      }
      mb_resolve_blocks(dt, blocks, numeric_only = TRUE, non_constant = FALSE)
    },

    .method_for_block = function(method_map, bn) {
      if (is.null(method_map)) {
        return("partial_corr")
      }
      m = method_map[[bn]]
      if (is.null(m)) "partial_corr" else as.character(m)
    },

    .validate_named_map = function(value, name, block_names) {
      if (is.null(value) || (is.list(value) && !length(value))) {
        return(invisible(TRUE))
      }
      if (!is.list(value) || is.null(names(value)) ||
        anyNA(names(value)) || any(!nzchar(names(value))) ||
        anyDuplicated(names(value))) {
        stop(sprintf(
          "`%s` must be a named list with unique, non-empty block names.",
          name
        ), call. = FALSE)
      }
      unknown = setdiff(names(value), block_names)
      if (length(unknown)) {
        stop(sprintf(
          "`%s` refers to unknown or unusable blocks: %s.",
          name,
          paste(unknown, collapse = ", ")
        ), call. = FALSE)
      }
      invisible(TRUE)
    },

    .validate_subgroup = function(subgroup, task) {
      n = task$nrow
      if (is.null(subgroup)) {
        return(rep.int(TRUE, n))
      }
      if (is.logical(subgroup)) {
        # Positional: only meaningful for the exact task passed to $train().
        if (length(subgroup) != n || anyNA(subgroup)) {
          stop(paste0(
            "A logical `subgroup` must be a non-missing vector with one entry per training row. ",
            "It is positional and not resampling-safe; use task row ids or a column specification instead."
          ), call. = FALSE)
        }
        selected = subgroup
      } else if (is.numeric(subgroup)) {
        if (!length(subgroup) || anyNA(subgroup) ||
          any(!is.finite(subgroup)) || any(subgroup != floor(subgroup))) {
          stop(
            "Numeric `subgroup` entries must be finite, integer-valued task row ids.",
            call. = FALSE
          )
        }
        # Ids that do not exist in the backend at all cannot be row ids of this
        # data (e.g. positions of a task whose row ids are not 1..n).
        unknown = setdiff(subgroup, task$backend$rownames)
        if (length(unknown)) {
          stop(sprintf(paste0(
            "`subgroup` contains %d id(s) that do not exist in the task's backend (e.g. %s). ",
            "Numeric `subgroup` entries must be task row ids (`task$row_ids`), not row positions."
          ), length(unknown), paste(utils::head(unknown, 3L), collapse = ", ")), call. = FALSE)
        }
        # Row ids travel with the data through filtering and resampling; ids of
        # rows outside the current training task are not selected.
        selected = task$row_ids %in% subgroup
      } else if (is.character(subgroup) || is.list(subgroup)) {
        spec = if (is.list(subgroup)) subgroup else list(column = subgroup, values = TRUE)
        column = spec$column
        if (!is.character(column) || length(column) != 1L || is.na(column) ||
          !nzchar(column) || !column %in% task$col_info$id) {
          stop(
            "`subgroup` must name exactly one column of the training task.",
            call. = FALSE
          )
        }
        if (is.list(subgroup) && (!is.atomic(spec$values) ||
          !length(spec$values) || anyNA(spec$values))) {
          stop(
            "`subgroup$values` must be a non-empty vector without missing values.",
            call. = FALSE
          )
        }
        value = task$data(rows = task$row_ids, cols = column)[[1L]]
        if (anyNA(value)) {
          stop(sprintf("`subgroup` column '%s' contains missing values.", column),
            call. = FALSE)
        }
        if (!is.list(subgroup) && !is.logical(value)) {
          stop(sprintf(paste0(
            "`subgroup` column '%s' must be logical; use ",
            "list(column = '%s', values = ...) for other column types."
          ), column, column), call. = FALSE)
        }
        selected = as.character(value) %in% as.character(spec$values)
      } else {
        stop(
          "`subgroup` must be NULL, task row ids, a logical column name, list(column = , values = ), or a logical vector.",
          call. = FALSE
        )
      }
      if (sum(selected) < 2L) {
        stop(sprintf(
          "`subgroup` must select at least two training rows (%d of %d selected).",
          sum(selected), n
        ), call. = FALSE)
      }
      selected
    },

    .prepare_design_data = function(data, schema = NULL, context, hint = NULL) {
      data = as.data.frame(data, stringsAsFactors = FALSE)
      if (!ncol(data) || anyNA(data)) {
        stop(sprintf(
          "%s must contain at least one complete covariate column.",
          context
        ), call. = FALSE)
      }

      if (is.null(schema)) {
        schema = lapply(names(data), function(name) {
          value = data[[name]]
          if (is.character(value) || is.factor(value) || is.logical(value)) {
            observed = unique(as.character(value))
            levels = if (is.factor(value)) {
              levels(value)[levels(value) %in% observed]
            } else {
              sort(observed)
            }
            if (length(levels) < 2L) {
              stop(sprintf(
                "%s categorical covariate '%s' must have at least two observed levels.",
                context,
                name
              ), call. = FALSE)
            }
            list(type = "factor", levels = levels, ordered = is.ordered(value))
          } else if (is.numeric(value)) {
            if (any(!is.finite(value))) {
              stop(sprintf("%s covariate '%s' must be finite.", context, name),
                call. = FALSE)
            }
            list(type = "numeric")
          } else {
            stop(sprintf(
              "%s covariate '%s' has unsupported class '%s'.",
              context,
              name,
              class(value)[[1L]]
            ), call. = FALSE)
          }
        }) |>
          stats::setNames(names(data))
      } else {
        if (!is.list(schema) || !identical(names(schema), names(data))) {
          stop(sprintf("%s does not match the fitted covariate schema.", context),
            call. = FALSE)
        }
      }

      for (name in names(schema)) {
        specification = schema[[name]]
        if (identical(specification$type, "numeric")) {
          if (!is.numeric(data[[name]]) || any(!is.finite(data[[name]]))) {
            stop(sprintf("%s covariate '%s' must be finite numeric data.",
              context, name), call. = FALSE)
          }
        } else if (identical(specification$type, "factor")) {
          values = as.character(data[[name]])
          unseen = setdiff(unique(values), specification$levels)
          if (length(unseen)) {
            stop(sprintf(
              "%s covariate '%s' contains unseen levels: %s.%s",
              context,
              name,
              paste(unseen, collapse = ", "),
              if (is.null(hint)) "" else paste0(" ", hint)
            ), call. = FALSE)
          }
          data[[name]] = factor(
            values,
            levels = specification$levels,
            ordered = isTRUE(specification$ordered)
          )
        } else {
          stop(sprintf("%s contains an invalid fitted covariate schema.",
            context), call. = FALSE)
        }
      }

      list(data = data, schema = schema)
    },

    .encode_site = function(site_vec, known_levels, strategy = c("other", "baseline"), ref_level = NULL) {
      strategy = match.arg(strategy)
      s = as.character(site_vec)
      unseen = is.na(s) | !s %in% known_levels
      baseline = if (is.null(ref_level)) known_levels[[1L]] else ref_level
      s[unseen] = baseline
      # Explicit treatment coding freezes the reference level and is independent
      # of global contrast options or literal site labels such as '.other'.
      mm = matrix(1, nrow = length(s), ncol = length(known_levels),
        dimnames = list(NULL, c("(Intercept)", paste0("site:", known_levels[-1L]))))
      for (j in seq_along(known_levels)[-1L]) {
        mm[, j] = as.numeric(s == known_levels[[j]])
      }
      if (identical(strategy, "other")) mm[unseen, ] = 0
      mm
    },

    # Row-wise median of a numeric matrix without an R-level loop over rows.
    .row_medians = function(M) {
      k = ncol(M)
      if (k == 1L) {
        return(M[, 1L])
      }
      sorted = matrix(M[order(row(M), M)], nrow = nrow(M), byrow = TRUE)
      if (k %% 2L) {
        sorted[, (k + 1L) %/% 2L]
      } else {
        (sorted[, k %/% 2L] + sorted[, k %/% 2L + 1L]) / 2
      }
    },

    # Geometric repair maps (Feldman et al., 2015) fitted on training rows:
    # x -> (1 - lambda) * x + lambda * Q_med(F_g(x)), with F_g the interpolated
    # empirical CDF of group g (inverse of type-7 quantiles, mid-ranks for ties)
    # and Q_med the pointwise median of the groups' type-7 quantile functions.
    # Each map stores the group's distinct training values and repair shifts.
    .fit_dir_maps = function(X, protected, lambda) {
      groups = levels(protected)
      maps = lapply(colnames(X), function(name) {
        sorted = lapply(split(X[, name], protected), sort)
        probs = lapply(sorted, function(values) {
          (seq_along(values) - 1) / (length(values) - 1)
        })
        group_maps = lapply(groups, function(group) {
          x = unique(sorted[[group]])
          if (lambda == 0) {
            return(list(x = x, shift = rep(0, length(x))))
          }
          p = stats::approx(sorted[[group]], probs[[group]], xout = x,
            ties = mean)$y
          quantiles = matrix(vapply(groups, function(other) {
            stats::approx(probs[[other]], sorted[[other]], xout = p)$y
          }, numeric(length(p))), nrow = length(p))
          list(x = x, shift = lambda * (private$.row_medians(quantiles) - x))
        })
        stats::setNames(group_maps, groups)
      })
      stats::setNames(maps, colnames(X))
    },

    .apply_dir = function(X, protected, repair_maps) {
      for (name in colnames(X)) {
        for (group in unique(protected)) {
          selected = protected == group
          map = repair_maps[[name]][[group]]
          if (is.null(map) || !length(map$x) ||
            length(map$x) != length(map$shift) ||
            any(!is.finite(c(map$x, map$shift)))) {
            stop("DIR fitted repair map is missing or invalid.", call. = FALSE)
          }
          shift = if (length(map$x) == 1L) {
            rep(map$shift, sum(selected))
          } else {
            stats::approx(map$x, map$shift, xout = X[selected, name],
              rule = 2, ties = "ordered")$y
          }
          X[selected, name] = X[selected, name] + shift
        }
      }
      X
    },

    .combat_valid_batches = function(est) {
      labs = character(0)
      if (!is.null(est$gamma.hat) && !is.null(rownames(est$gamma.hat))) {
        labs = rownames(est$gamma.hat)
      } else if (!is.null(est$gamma.star) && !is.null(rownames(est$gamma.star))) {
        labs = rownames(est$gamma.star)
      } else if (!is.null(est$batch)) {
        labs = unique(as.character(est$batch))
      }
      unique(as.character(labs))
    },

    # ---- train --------------------------------------------------------------

    .train_task = function(task) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE)

      blocks = private$.validate_blocks(task, pv$blocks)
      blocks = Filter(length, blocks)
      if (!length(blocks)) {
        stop("PipeOpSiteCorrection: no numeric features remain in any block.",
          call. = FALSE)
      }
      private$.validate_named_map(
        pv$site_correction, "site_correction", names(blocks)
      )
      private$.validate_named_map(pv$method, "method", names(blocks))
      if (!is.null(pv$method)) {
        invalid_methods = vapply(pv$method, function(value) {
          !is.null(value) && (
            length(value) != 1L || !is.character(value) || is.na(value) ||
              !value %in% c("partial_corr", "combat", "dir")
          )
        }, logical(1L))
        if (any(invalid_methods)) {
          stop(sprintf(
            "Invalid site-correction methods for blocks: %s.",
            paste(names(pv$method)[invalid_methods], collapse = ", ")
          ), call. = FALSE)
        }
      }
      if (is.null(pv$site_correction) || !length(pv$site_correction)) {
        self$state = list(
          blocks = blocks,
          per_block = list(),
          unknown_site = pv$unknown_site,
          keep_site_col = isTRUE(pv$keep_site_col)
        )
        return(task)
      }

      # Collect all referenced site/covariate columns across blocks (for data pull)
      site_cols_used = character(0)
      for (bn in names(blocks)) {
        x = pv$site_correction[[bn]]
        if (is.null(x)) next
        m = private$.method_for_block(pv$method, bn)
        if (identical(m, "combat")) {
          if (is.list(x)) {
            site_cols_used = c(site_cols_used,
              as.character(x$site),
              as.character(x$covariates %||% character(0)))
          } else {
            site_cols_used = c(site_cols_used, as.character(x)[1L])
          }
        } else {
          site_cols_used = c(site_cols_used, as.character(x))
        }
      }
      site_cols_used = unique(site_cols_used[nzchar(site_cols_used)])

      cols_needed = intersect(
        unique(c(task$feature_names, site_cols_used)),
        task$col_info$id
      )
      dt = task$data(rows = task$row_ids, cols = cols_needed)

      have_neuro = requireNamespace("neuroCombat", quietly = TRUE)
      per_block = list()

      if (!is.null(pv$subgroup)) {
        corrected = names(blocks)[vapply(names(blocks), function(bn) {
          !is.null(pv$site_correction[[bn]])
        }, logical(1L))]
        uses_partial = vapply(corrected, function(bn) {
          identical(private$.method_for_block(pv$method, bn), "partial_corr")
        }, logical(1L))
        if (!any(uses_partial)) {
          stop(
            "`subgroup` is only used by 'partial_corr' blocks, but no corrected block uses 'partial_corr'.",
            call. = FALSE
          )
        }
      }
      idx_fit = private$.validate_subgroup(pv$subgroup, task)
      target_cols = task$target_names

      blocks_eff = blocks # will hold features *excluding* any referenced site/covariate columns

      for (bn in names(blocks)) {
        xspec = pv$site_correction[[bn]]
        if (is.null(xspec)) next

        method = private$.method_for_block(pv$method, bn)

        # --- parse site spec per method
        site_cols = character(0)
        combat_site = NULL
        combat_covs = character(0)

        if (identical(method, "combat")) {
          if (is.list(xspec)) {
            combat_site = as.character(xspec$site)
            if (length(combat_site) != 1L || !nzchar(combat_site)) {
              stop(sprintf("Block '%s' (combat): 'site' must be character(1).", bn))
            }
            combat_covs = as.character(xspec$covariates %||% character(0))
            if (combat_site %in% combat_covs) {
              stop(sprintf(
                "Block '%s' (combat): the batch column must not also be a covariate.",
                bn
              ), call. = FALSE)
            }
          } else {
            # backward compat: single string is the site
            # Emit a one-time warning so users know to update to list() format
            warning(sprintf(
              "Block '%s' (combat): passing a bare character string as site specification is deprecated. Use list(site = \"%s\", covariates = character(0)) for explicit and unambiguous specification.",
              bn, xspec[1L]
            ), call. = FALSE)
            xs = as.character(xspec)
            if (!length(xs)) next
            combat_site = xs[1L]
            combat_covs = character(0)
          }
          site_cols = unique(c(combat_site, combat_covs))
        } else {
          site_cols = as.character(xspec)
        }

        if (!length(site_cols) || anyNA(site_cols) || any(!nzchar(site_cols))) {
          stop(sprintf(
            "Block '%s': site/covariate specification must be non-empty and non-missing.",
            bn
          ), call. = FALSE)
        }
        site_cols = unique(site_cols)
        missing_sites = setdiff(site_cols, names(dt))
        if (length(missing_sites)) {
          stop(sprintf("Block '%s': missing referenced column(s): %s", bn, paste(missing_sites, collapse = ", ")))
        }
        # Columns read at prediction time must not be targets; ComBat covariates
        # are only used for training and may reference the target.
        leak = intersect(
          if (identical(method, "combat")) combat_site else site_cols,
          target_cols
        )
        if (length(leak)) {
          stop(sprintf(paste0(
            "Block '%s' (%s): target column(s) %s cannot be used as site, protected ",
            "or partial-correlation columns because they are read at prediction time; ",
            "only ComBat `covariates` may reference a target (training only)."
          ), bn, method, paste(leak, collapse = ", ")), call. = FALSE)
        }
        if (anyNA(dt[, .SD, .SDcols = site_cols])) {
          stop(sprintf(
            "Block '%s': referenced site/covariate columns contain missing values.",
            bn
          ), call. = FALSE)
        }

        Xcols_raw = blocks[[bn]]
        # remove any referenced site/covariate columns from the features of this block
        Xcols = setdiff(Xcols_raw, site_cols)
        if (!identical(Xcols_raw, Xcols)) {
          lgr$info("Block '%s': dropping %d referenced column(s) from features: %s",
            bn, length(setdiff(Xcols_raw, Xcols)), paste(setdiff(Xcols_raw, Xcols), collapse = ", "))
        }
        if (!length(Xcols)) {
          lgr$info("Block '%s': no non-site features left after exclusion; skipping correction", bn)
          next
        }
        X = .mb_numeric_matrix(
          as.matrix(dt[, .SD, .SDcols = Xcols]),
          sprintf("PipeOpSiteCorrection training block '%s'", bn)
        )
        colnames(X) = Xcols

        if (identical(method, "partial_corr")) {

          single_cat = length(site_cols) == 1L && (is.factor(dt[[site_cols]]) || is.character(dt[[site_cols]]))
          if (single_cat) {
            site_vec = dt[[site_cols]]
            site_lvls = levels(factor(site_vec))
            if (length(site_lvls) < 2L) {
              stop(sprintf(
                "Block '%s' (partial_corr): categorical site '%s' must have at least two observed levels.",
                bn,
                site_cols
              ), call. = FALSE)
            }
            G_all = private$.encode_site(site_vec, site_lvls, strategy = "other")
            G_fit = G_all[idx_fit, , drop = FALSE]
            design_kind = "categorical"
            design_schema = NULL
            design_contrasts = NULL
          } else {
            Z_all = dt[, .SD, .SDcols = site_cols]
            prepared = private$.prepare_design_data(
              Z_all,
              context = sprintf("Block '%s' (partial_corr) training design", bn)
            )
            G_all = stats::model.matrix(
              ~.,
              data = prepared$data,
              na.action = stats::na.fail
            )
            G_fit = G_all[idx_fit, , drop = FALSE]
            site_lvls = NULL
            design_kind = "matrix"
            design_schema = prepared$schema
            design_contrasts = attr(G_all, "contrasts")
          }
          design_cols = colnames(G_fit)
          if (nrow(G_fit) < ncol(G_fit) && !(pv$regularization > 0)) {
            stop(sprintf(
              paste0(
                "Block '%s' (partial_corr): the selected subgroup has fewer ",
                "rows (%d) than design columns (%d); use a positive ",
                "regularization value or a larger subgroup."
              ),
              bn,
              nrow(G_fit),
              ncol(G_fit)
            ), call. = FALSE)
          }

          lambda = pv$regularization %||% 0
          if (lambda == 0) {
            qr_fit = qr(G_fit)
            if (qr_fit$rank < ncol(G_fit)) {
              stop(sprintf(
                paste0(
                  "Block '%s' (partial_corr): the training design is rank deficient; ",
                  "remove redundant covariates, include every site in the fitting ",
                  "subgroup, or use positive regularization."
                ), bn
              ), call. = FALSE)
            }
            # Rows with leverage one (e.g. the only fitting row of a site) are
            # reproduced exactly, so their residuals vanish.
            leverage = rowSums(qr.Q(qr_fit)^2)
            exact = leverage > 1 - sqrt(.Machine$double.eps)
            if (any(exact)) {
              categorical = if (identical(design_kind, "categorical")) {
                stats::setNames(list(factor(site_vec, levels = site_lvls)), site_cols)
              } else {
                Filter(is.factor, prepared$data)
              }
              singletons = unlist(lapply(names(categorical), function(name) {
                counts = table(categorical[[name]][idx_fit])
                single = names(counts)[counts == 1L]
                if (length(single)) paste0(name, " = ", single) else character(0)
              }), use.names = FALSE)
              detail = if (length(singletons)) {
                sprintf(" (levels with a single fitting row: %s)",
                  paste(singletons, collapse = ", "))
              } else {
                ""
              }
              warning(sprintf(paste0(
                "Block '%s' (partial_corr): %d fitting row(s) are reproduced exactly by the ",
                "unpenalized site design%s. Their corrected values keep no within-site ",
                "information (they equal the training grand means unless zero_center or ",
                "revertflag is set), and the corresponding site offsets rest on a single ",
                "observation. Consider regularization > 0 or merging small sites."
              ), bn, sum(exact), detail), call. = FALSE)
            }
          }
          if (lambda > 0) {
            beta = cpp_lm_coeff_ridge(
              as.matrix(G_fit),
              as.matrix(X[idx_fit, , drop = FALSE]),
              lambda,
              which(colnames(G_fit) %in% "(Intercept)")
            )
          } else {
            beta = cpp_lm_coeff(
              as.matrix(G_fit),
              as.matrix(X[idx_fit, , drop = FALSE])
            )
          }

          rownames(beta) = colnames(G_fit)
          colnames(beta) = colnames(X)

          mu = colMeans(X, na.rm = TRUE)
          Xcorr = if (isTRUE(pv$revertflag)) X + G_all %*% beta else X - G_all %*% beta
          if (!isTRUE(pv$zero_center)) {
            Xcorr = sweep(Xcorr, 2, mu, "+")
          }
          dt[, (Xcols) := as.data.table(Xcorr)]

          per_block[[bn]] = list(
            method = "partial_corr",
            site_cols = site_cols,
            design_cols = design_cols,
            beta = beta,
            means = mu,
            zero_center = isTRUE(pv$zero_center),
            revert = isTRUE(pv$revertflag),
            design_kind = design_kind,
            site_lvls = site_lvls,
            design_schema = design_schema,
            design_contrasts = design_contrasts
          )
          blocks_eff[[bn]] = Xcols

        } else if (identical(method, "combat")) {

          if (!have_neuro) .sitecorr_neurocombat_missing("ComBat")

          # Build mod from covariates (if any), using only levels observed in
          # the training rows.
          if (length(combat_covs)) {
            prepared_cov = private$.prepare_design_data(
              dt[, .SD, .SDcols = combat_covs],
              context = sprintf("Block '%s' (combat) covariate design", bn)
            )
            mod_mat = stats::model.matrix(~., data = prepared_cov$data)
          } else {
            mod_mat = NULL
          }

          # Backend factor levels without training rows (e.g. a held-out site)
          # would create empty batches.
          site_vec = droplevels(factor(dt[[combat_site]]))
          if (nlevels(site_vec) < 2L) {
            stop(sprintf(
              "Block '%s' (combat): batch column '%s' must have at least two observed levels in the training rows.",
              bn, combat_site
            ), call. = FALSE)
          }
          if (!is.null(pv$ref_batch)) {
            if (length(pv$ref_batch) != 1L || is.na(pv$ref_batch)) {
              stop("`ref_batch` must be NULL or a single batch label.", call. = FALSE)
            }
            if (!as.character(pv$ref_batch) %in% levels(site_vec)) {
              stop(sprintf(
                "Block '%s' (combat): ref_batch '%s' has no training rows.",
                bn, as.character(pv$ref_batch)
              ), call. = FALSE)
            }
          }
          res = neuroCombat::neuroCombat(
            dat = t(X),
            batch = site_vec,
            mod = mod_mat,
            eb = pv$eb %||% TRUE,
            parametric = TRUE,
            mean.only = pv$mean_only %||% FALSE,
            ref.batch = if (is.null(pv$ref_batch)) NULL else as.character(pv$ref_batch),
            verbose = pv$verbose %||% FALSE
          )
          dt[, (Xcols) := as.data.table(t(res$dat.combat))]

          valid_batches = private$.combat_valid_batches(res$estimates)
          per_block[[bn]] = list(
            method     = "combat",
            site_cols  = site_cols, # union(site_var, covariates)
            site_var   = combat_site, # single batch column
            covariates = combat_covs, # character()
            site_lvls  = valid_batches,
            estimates  = res$estimates,
            ref_batch  = if (!is.null(pv$ref_batch) && pv$ref_batch %in% valid_batches) pv$ref_batch else valid_batches[1]
          )
          blocks_eff[[bn]] = Xcols

        } else if (identical(method, "dir")) {

          if (length(site_cols) != 1L) {
            stop(sprintf("Block '%s' (dir): exactly one categorical column required.", bn))
          }

          prot_vec = factor(dt[[site_cols]])
          if (nlevels(prot_vec) < 2L) {
            stop(sprintf(
              "Block '%s' (dir): protected attribute '%s' must have at least 2 levels, but only %d level(s) found in the data.",
              bn, site_cols, nlevels(prot_vec)
            ), call. = FALSE)
          }
          lambda = pv$lambda %||% 0.5
          # The group distribution functions are undefined for a feature that is
          # constant within a group.
          if (lambda > 0) {
            degenerate = vapply(Xcols, function(name) {
              any(vapply(split(X[, name], prot_vec), function(values) {
                length(unique(values)) < 2L
              }, logical(1L)))
            }, logical(1L))
            if (any(degenerate)) {
              stop(sprintf(
                "Block '%s' (dir): each protected group needs at least two distinct training values for features: %s.",
                bn, paste(Xcols[degenerate], collapse = ", ")
              ), call. = FALSE)
            }
          }
          repair_maps = private$.fit_dir_maps(X, prot_vec, lambda)
          repaired = private$.apply_dir(X, as.character(prot_vec), repair_maps)
          dt[, (Xcols) := as.data.table(repaired)]

          per_block[[bn]] = list(
            method = "dir",
            site_cols = site_cols,
            site_lvls = levels(prot_vec),
            lambda = lambda,
            repair_maps = repair_maps
          )
          blocks_eff[[bn]] = Xcols

        } else {
          stop(sprintf("Unknown method '%s' for block '%s'", method, bn))
        }
      }

      # Replace corrected features in place so that column information, keys
      # and roles stay consistent with the backend.
      keep_site = pv$keep_site_col %||% FALSE
      all_site_cols = unique(unlist(lapply(per_block, `[[`, "site_cols"), use.names = FALSE))
      corrected_cols = unique(unlist(blocks_eff[names(per_block)], use.names = FALSE))
      mb_task_replace_features(task, dt,
        changed = corrected_cols,
        drop = if (keep_site) character(0) else all_site_cols
      )

      self$state = list(
        blocks        = blocks_eff,
        per_block     = per_block,
        unknown_site  = pv$unknown_site,
        keep_site_col = keep_site
      )

      lgr$info(
        "SiteCorr: %d/%d block(s) corrected | %s | policies: unknown_site=%s, combat_unknown=%s, keep_site_col=%s",
        length(self$state$per_block),
        length(self$state$blocks),
        paste(vapply(names(self$state$per_block), function(bn) {
          info = self$state$per_block[[bn]]
          featsN = length(self$state$blocks[[bn]])
          paste0(
            bn, "{", info$method,
            "; feats=", featsN,
            "; site=", paste(info$site_cols, collapse = ","),
            if (!is.null(info$ref_batch)) paste0("; ref=", info$ref_batch) else "",
            "}"
          )
        }, character(1)), collapse = "; "),
        if (is.null(self$state$unknown_site)) "other" else self$state$unknown_site,
        {
          v = self$param_set$get_values(tags = "predict")$combat_unknown
          if (is.null(v)) {
            v = self$param_set$get_values(tags = "train")$combat_unknown
          }
          if (is.null(v)) "noop" else v
        },
        as.character(isTRUE(self$state$keep_site_col))
      )

      task
    },

    # ---- predict ------------------------------------------------------------

    .predict_task = function(task) {
      st = self$state
      if (is.null(st) || is.null(st$blocks) || !length(st$blocks)) {
        return(task)
      }
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "predict"),
        keep.null = TRUE)

      site_cols_needed = unique(unlist(lapply(st$per_block, function(info) {
        if (identical(info$method, "combat")) {
          info$site_var
        } else {
          info$site_cols
        }
      }), use.names = FALSE))
      leak = intersect(site_cols_needed, task$target_names)
      if (length(leak)) {
        stop(sprintf(
          "[%s] Target column(s) %s cannot be read as site or protected columns at prediction time.",
          self$id, paste(leak, collapse = ", ")
        ), call. = FALSE)
      }
      missing_sites = setdiff(site_cols_needed, task$col_info$id)
      if (length(missing_sites)) {
        stop(sprintf(
          "[%s] Prediction task lacks site or protected column(s): %s.",
          self$id, paste(missing_sites, collapse = ", ")
        ), call. = FALSE)
      }
      cols_needed = unique(c(task$feature_names, site_cols_needed))
      dt = task$data(rows = task$row_ids, cols = cols_needed)

      trained_feats = unique(unlist(st$blocks %||% list(), use.names = FALSE))
      if (length(trained_feats)) {
        mb_assert_columns_present(
          colnames_dt = names(dt),
          required = trained_feats,
          context = sprintf("[%s] Prediction task", self$id),
          hint = "Apply the same preprocessing used during training and retain all trained feature columns before PipeOpSiteCorrection."
        )
      }

      unknown_strategy = pv$unknown_site %||% st$unknown_site %||% "other"
      combat_policy = pv$combat_unknown %||% "noop"

      for (bn in names(st$blocks)) {
        info = st$per_block[[bn]]
        if (is.null(info)) next

        Xcols = st$blocks[[bn]]
        X = .mb_numeric_matrix(
          as.matrix(dt[, .SD, .SDcols = Xcols]),
          sprintf("PipeOpSiteCorrection prediction block '%s'", bn)
        )
        colnames(X) = Xcols

        if (identical(info$method, "partial_corr")) {
          # rebuild design
          if (identical(info$design_kind, "categorical")) {
            site_vec = dt[[info$site_cols]]
            s = as.character(site_vec)
            unseen_mask = !(s %in% info$site_lvls) | is.na(s)
            G = private$.encode_site(site_vec, info$site_lvls,
              strategy = ifelse(unknown_strategy == "baseline", "baseline", "other"))
            add = setdiff(info$design_cols, colnames(G))
            if (length(add) > 0L) {
              G = cbind(G, matrix(0, nrow(G), length(add), dimnames = list(NULL, add)))
            }
            drop = setdiff(colnames(G), info$design_cols)
            if (length(drop) > 0L) {
              G = G[, setdiff(colnames(G), drop), drop = FALSE]
            }
            G = G[, info$design_cols, drop = FALSE]
            if (any(unseen_mask) && !identical(unknown_strategy, "baseline")) {
              G[unseen_mask, ] = 0
            }
          } else {
            Z_new = dt[, .SD, .SDcols = info$site_cols]
            prepared = private$.prepare_design_data(
              Z_new,
              schema = info$design_schema,
              context = sprintf("Block '%s' (partial_corr) prediction design", bn),
              hint = "`unknown_site` applies only to a single categorical site column."
            )
            G = stats::model.matrix(
              ~.,
              data = prepared$data,
              contrasts.arg = info$design_contrasts,
              na.action = stats::na.fail
            )
            add = setdiff(info$design_cols, colnames(G))
            if (length(add) > 0L) {
              G = cbind(G, matrix(0, nrow(G), length(add), dimnames = list(NULL, add)))
            }
            drop = setdiff(colnames(G), info$design_cols)
            if (length(drop) > 0L) {
              G = G[, setdiff(colnames(G), drop), drop = FALSE]
            }
            G = G[, info$design_cols, drop = FALSE]
          }

          if (nrow(G) != nrow(X)) {
            stop(sprintf(
              "Block '%s' (partial_corr): design matrix rows (%d) do not match feature matrix rows (%d).",
              bn,
              nrow(G),
              nrow(X)
            ), call. = FALSE)
          }

          beta = info$beta
          if (!is.matrix(beta) || !is.numeric(beta) || anyNA(beta) ||
            any(!is.finite(beta)) || is.null(rownames(beta)) ||
            is.null(colnames(beta)) || anyNA(rownames(beta)) ||
            anyNA(colnames(beta)) || any(!nzchar(rownames(beta))) ||
            any(!nzchar(colnames(beta))) || anyDuplicated(rownames(beta)) ||
            anyDuplicated(colnames(beta))) {
            stop(sprintf(
              "Block '%s' (partial_corr): stored coefficients are invalid.",
              bn
            ), call. = FALSE)
          }
          miss_r = setdiff(colnames(G), rownames(beta))
          extra_r = setdiff(rownames(beta), colnames(G))
          if (length(miss_r) || length(extra_r)) {
            stop(sprintf(
              paste0(
                "Block '%s' (partial_corr): stored coefficient rows do not ",
                "match the fitted design. Missing: [%s]; unexpected: [%s]."
              ),
              bn,
              if (length(miss_r)) mb_format_truncated(miss_r) else "none",
              if (length(extra_r)) mb_format_truncated(extra_r) else "none"
            ), call. = FALSE)
          }
          beta = beta[colnames(G), , drop = FALSE]

          miss_c = setdiff(colnames(beta), colnames(X))
          extra_c = setdiff(colnames(X), colnames(beta))
          if (length(miss_c) || length(extra_c)) {
            stop(sprintf(
              "Block '%s' (partial_corr): trained feature set and stored coefficient columns do not match. Missing in new data/state alignment: [%s]; unexpected columns: [%s].",
              bn,
              if (length(miss_c)) mb_format_truncated(miss_c) else "none",
              if (length(extra_c)) mb_format_truncated(extra_c) else "none"
            ), call. = FALSE)
          }
          X = X[, colnames(beta), drop = FALSE]

          GB = G %*% beta
          if (!identical(colnames(GB), colnames(X))) {
            stop(sprintf(
              "Block '%s' (partial_corr): corrected design output columns do not match the trained feature columns.",
              bn
            ), call. = FALSE)
          }

          means = mb_align_named_numeric(
            info$means,
            cols = colnames(X),
            context = sprintf(
              "PipeOpSiteCorrection fitted means for block '%s'", bn
            )
          )
          Xcorr = if (isTRUE(info$revert)) X + GB else X - GB
          if (!isTRUE(info$zero_center)) {
            Xcorr = sweep(Xcorr, 2, means, "+")
          }
          # Unseen sites receive no site-specific correction but share the
          # output location of corrected rows (centred when zero_center = TRUE).
          if (identical(info$design_kind, "categorical") &&
            !identical(unknown_strategy, "baseline") && any(unseen_mask)) {
            Xunseen = X[unseen_mask, , drop = FALSE]
            if (isTRUE(info$zero_center)) {
              Xunseen = sweep(Xunseen, 2, means, "-")
            }
            Xcorr[unseen_mask, ] = Xunseen
          }
          dt[, (Xcols) := as.data.table(Xcorr)]

        } else if (identical(info$method, "combat")) {
          if (!requireNamespace("neuroCombat", quietly = TRUE)) .sitecorr_neurocombat_missing("ComBat prediction")

          # Warn if training used covariates: neuroCombatFromTraining does not apply
          # the covariate model to new data, so predict-time covariate effects are
          # approximated from training estimates only.
          if (length(info$covariates) > 0L) {
            warning(sprintf(
              "Block '%s' (combat): model was trained with covariate(s) [%s], but neuroCombatFromTraining() does not re-apply the covariate model at predict time. Batch effect removal at prediction may differ from training.",
              bn, paste(info$covariates, collapse = ", ")
            ), call. = FALSE)
          }

          valid = private$.combat_valid_batches(info$estimates)
          if (!length(valid)) {
            dt[, (Xcols) := as.data.table(X)]
            next
          }

          # use only the single site var for batch at predict time
          batch_chr = as.character(dt[[info$site_var]])
          known_mask = (batch_chr %in% valid) & !is.na(batch_chr)

          if (identical(combat_policy, "noop")) {
            idx = which(known_mask)
            # Write whole numeric columns so the output type does not depend on
            # which rows belong to known batches.
            if (!length(idx)) {
              dt[, (Xcols) := as.data.table(X)]
              next
            }
            Yk = t(X[idx, , drop = FALSE])
            bk = factor(batch_chr[idx], levels = valid)
            Yck = tryCatch(
              neuroCombat::neuroCombatFromTraining(
                dat = Yk, batch = bk, estimates = info$estimates
              )$dat.combat,
              error = function(e) {
                stop(sprintf(
                  "Block '%s' (combat): neuroCombatFromTraining failed for known batches: %s",
                  bn,
                  conditionMessage(e)
                ), call. = FALSE)
              }
            )
            X[idx, ] = t(Yck)
            dt[, (Xcols) := as.data.table(X)]
          } else {
            baseline = if (!is.null(info$ref_batch) && info$ref_batch %in% valid) info$ref_batch else valid[1]
            bmap = batch_chr
            bmap[!known_mask | is.na(bmap)] = baseline
            Y = t(X)
            Yc = tryCatch(
              neuroCombat::neuroCombatFromTraining(
                dat = Y, batch = factor(bmap, levels = valid), estimates = info$estimates
              )$dat.combat,
              error = function(e) {
                stop(sprintf(
                  "Block '%s' (combat): neuroCombatFromTraining failed under combat_unknown='%s': %s",
                  bn,
                  combat_policy,
                  conditionMessage(e)
                ), call. = FALSE)
              }
            )
            dt[, (Xcols) := as.data.table(t(Yc))]
          }

        } else if (identical(info$method, "dir")) {
          protected = as.character(dt[[info$site_cols]])
          unseen = unique(protected[is.na(protected) |
            !protected %in% info$site_lvls])
          if (length(unseen)) {
            stop(sprintf(
              "Block '%s' (dir): prediction contains missing or unseen protected levels: %s.",
              bn,
              paste(unseen, collapse = ", ")
            ), call. = FALSE)
          }
          repaired = private$.apply_dir(X, protected, info$repair_maps)
          dt[, (Xcols) := as.data.table(repaired)]
        }
      }

      keep_site = pv$keep_site_col %||% FALSE
      all_site_cols = unique(unlist(lapply(st$per_block, `[[`, "site_cols"), use.names = FALSE))
      corrected_cols = unique(unlist(st$blocks[names(st$per_block)], use.names = FALSE))
      mb_task_replace_features(task, dt,
        changed = corrected_cols,
        drop = if (keep_site) character(0) else all_site_cols
      )
      task
    }
  )
)

# neuroCombat is distributed on GitHub only; point users to the revision used
# in continuous integration.
.sitecorr_neurocombat_missing = function(context) {
  stop(sprintf(
    paste0(
      "%s requires the GitHub-only package 'neuroCombat'. Install it with ",
      "remotes::install_github(",
      "\"Jfortin1/neuroCombat_Rpackage@fbec46a61bc92bedb450b0e44addae4ce6afa934\")."
    ),
    context
  ), call. = FALSE)
}
