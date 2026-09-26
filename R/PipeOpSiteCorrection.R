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
#' @details
#' **Partial correlation (`"partial_corr"`)**
#' - If the site spec is a **single categorical** column, we build a dummy-coded
#'   design with an intercept and keep a stable column layout across train/predict.
#'   At predict-time, **unseen site labels** are treated as **no-op rows** (i.e.,
#'   zero design contribution) or can be mapped to baseline if you set
#'   `unknown_site = "baseline"`.
#' - If the site spec is **multiple columns** (e.g., PRS PCs) or numeric,
#'   we construct the design as `cbind(1, Z)` via `model.matrix(~ .)`.
#' - We solve a ridge-stabilized normal equation for the site effects and
#'   subtract them; optional mean re-add (`zero_center = FALSE` by default).
#'
#' **ComBat (`"combat"`, via \pkg{neuroCombat}) - now with `mod` support**
#' - Trains using `neuroCombat(dat = t(X), batch = site, mod = MM, ...)`, where
#'   `MM = model.matrix(~ ., data = covariates)`; character covariates are
#'   auto-factorized. ComBat accepts **one** batch vector, but **many covariates**.
#'   We store the returned `estimates`, the valid batch levels, the `site_var`,
#'   and the list of `covariates`.
#' - At predict, we apply `neuroCombatFromTraining(dat, batch, estimates)`.
#'   The upstream function **does not support** supplying `mod` for new data;
#'   if estimates were trained with `mod` (i.e., `covariates` is non-empty),
#'   the batch-effect removal at predict-time uses the training covariate-effect
#'   estimates only and does NOT re-apply the covariate model to new observations.
#'   **A `warning()` is emitted at predict-time** whenever `covariates` is non-empty
#'   to alert users of this limitation. If predict-time covariate correction matters,
#'   consider using `"partial_corr"` instead.
#'   Unseen batches can be handled with
#'   `combat_unknown = "noop"` (skip) or `"baseline"` (map to `ref_batch`).
#'
#' **DIR (`"dir"`, via \pkg{fairmodels})**
#' - Applies distribution repair per block with the given `lambda`.
#' - Prediction uses feature- and group-specific repair maps fitted on training
#'   observations, with linear interpolation between observed values and constant
#'   repair shifts beyond the training range. It never re-estimates distributions
#'   from prediction data. Missing or unseen protected groups are rejected.
#'
#' The operator preserves targets, reconstructs a backend with stable row ids,
#' and (by default) **drops** all site/covariate columns referenced in `site_correction`
#' unless `keep_site_col = TRUE`.
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
#'         unseen **categorical** site under `"partial_corr"`. Default `"other"`.
#'       \item `zero_center` (`logical(1)`): Re-add grand means? Default `FALSE`.
#'       \item `revertflag` (`logical(1)`): Add instead of subtract the site effect.
#'       \item `regularization` (`numeric(1)`): Ridge penalty to stabilize the site
#'         regression. A value of `0` requests unpenalized least squares.
#'       \item `subgroup` (`logical()`/`integer()`): Optional row subset for fitting.
#'     }
#'   }
#'   \item{**ComBat (neuroCombat)**}{
#'     \itemize{
#'       \item `eb` (`logical(1)`): Empirical Bayes shrinkage. Default `TRUE`.
#'       \item `mean_only` (`logical(1)`): Adjust means only. Default `FALSE`.
#'       \item `ref_batch` (`character(1)` or `NULL`): Optional reference.
#'       \item `combat_unknown` (`"noop"|"baseline"`): Predict-time policy for
#'         unseen batches. Default `"noop"`.
#'     }
#'   }
#'   \item{**DIR (fairmodels)**}{
#'     \itemize{
#'       \item `lambda` (`numeric(1)` in \[0,1\]): Repair strength. Default `0.5`.
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
#'       \item `lambda`, `site_lvls` (dir).
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
#'   [neuroCombat::neuroCombat] / [neuroCombat::neuroCombatFromTraining],
#'   [fairmodels::disparate_impact_remover].
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

    .validate_subgroup = function(subgroup, n) {
      if (is.null(subgroup)) {
        return(rep.int(TRUE, n))
      }
      if (is.logical(subgroup)) {
        if (length(subgroup) != n || anyNA(subgroup)) {
          stop(
            "`subgroup` must be a non-missing logical vector with one entry per training row.",
            call. = FALSE
          )
        }
        selected = subgroup
      } else if (is.numeric(subgroup)) {
        if (!length(subgroup) || anyNA(subgroup) ||
          any(!is.finite(subgroup)) || any(subgroup != floor(subgroup)) ||
          any(subgroup < 1L | subgroup > n)) {
          stop(
            "Numeric `subgroup` entries must be valid training-row positions.",
            call. = FALSE
          )
        }
        selected = seq_len(n) %in% as.integer(subgroup)
      } else {
        stop("`subgroup` must be NULL, logical, or integer-valued numeric.",
          call. = FALSE)
      }
      if (sum(selected) < 2L) {
        stop("`subgroup` must select at least two training rows.",
          call. = FALSE)
      }
      selected
    },

    .prepare_design_data = function(data, schema = NULL, context) {
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
              "%s covariate '%s' contains unseen levels: %s.",
              context,
              name,
              paste(unseen, collapse = ", ")
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

      cols_needed = unique(c(task$feature_names, site_cols_used))
      dt = task$data(rows = task$row_ids, cols = cols_needed)

      have_neuro = requireNamespace("neuroCombat", quietly = TRUE)
      have_fm = requireNamespace("fairmodels", quietly = TRUE)
      per_block = list()

      idx_fit = private$.validate_subgroup(pv$subgroup, nrow(dt))

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
          if (lambda == 0 && qr(G_fit)$rank < ncol(G_fit)) {
            stop(sprintf(
              paste0(
                "Block '%s' (partial_corr): the training design is rank deficient; ",
                "remove redundant covariates, include every site in the fitting ",
                "subgroup, or use positive regularization."
              ), bn
            ), call. = FALSE)
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

          if (!have_neuro) stop("ComBat requires 'neuroCombat'.")

          # Build mod from covariates (if any)
          if (length(combat_covs)) {
            Zcov = dt[, .SD, .SDcols = combat_covs]
            for (cc in names(Zcov)) if (is.character(Zcov[[cc]])) Zcov[, (cc) := factor(get(cc))]
            mod_mat = stats::model.matrix(~., data = Zcov)
          } else {
            mod_mat = NULL
          }

          site_vec = dt[[combat_site]]
          res = neuroCombat::neuroCombat(
            dat = t(X),
            batch = site_vec,
            mod = mod_mat,
            eb = pv$eb %||% TRUE,
            parametric = TRUE,
            mean.only = pv$mean_only %||% FALSE,
            ref.batch = pv$ref_batch,
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

          if (!have_fm) stop("DIR requires the 'fairmodels' package.")
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
          dat = data.frame(dt[, .SD, .SDcols = Xcols], protected = prot_vec)
          lambda = pv$lambda %||% 0.5
          # fairmodels quantizes even at lambda = 0 and cannot repair a feature
          # that is constant in any group; handle both cases explicitly.
          if (lambda > 0) {
            degenerate = vapply(Xcols, function(name) {
              any(vapply(split(dat[[name]], prot_vec), function(values) {
                length(unique(values)) < 2L
              }, logical(1L)))
            }, logical(1L))
            if (any(degenerate)) {
              stop(sprintf(
                "Block '%s' (dir): each protected group needs at least two distinct training values for features: %s.",
                bn, paste(Xcols[degenerate], collapse = ", ")
              ), call. = FALSE)
            }
            repaired = fairmodels::disparate_impact_remover(
              data = dat,
              protected = prot_vec,
              features_to_transform = Xcols,
              lambda = lambda
            )
          } else {
            repaired = dat
          }
          repair_maps = lapply(Xcols, function(name) {
            maps = lapply(levels(prot_vec), function(group) {
              idx = which(prot_vec == group)
              idx = idx[order(dat[[name]][idx])]
              idx = idx[!duplicated(dat[[name]][idx])]
              list(x = dat[[name]][idx],
                shift = repaired[[name]][idx] - dat[[name]][idx])
            })
            stats::setNames(maps, levels(prot_vec))
          })
          names(repair_maps) = Xcols
          dt[, (Xcols) := as.data.table(repaired[, Xcols, drop = FALSE])]

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

      out_dt = dt
      row_ids = task$row_ids
      pk_col = mb_make_backend_key_name(c(names(out_dt), task$col_info$id), "..row_id_sitecorr")
      out_dt[, (pk_col) := row_ids]

      # --- bring back all non-feature-role columns from the original task
      roles_orig = task$col_roles
      nonfeat_roles = setdiff(names(roles_orig), "feature")
      extra_cols = unique(unlist(roles_orig[nonfeat_roles], use.names = FALSE))
      extra_cols = setdiff(extra_cols, names(dt)) # avoid duplicates

      if (length(extra_cols)) {
        extra_dt = task$data(rows = task$row_ids, cols = extra_cols)
        dt_out = cbind(dt, extra_dt)
      } else {
        dt_out = dt
      }

      new_task = task$clone()
      new_task$backend = mlr3::as_data_backend(dt_out, primary_key = pk_col)

      # --- features (drop referenced columns from features if keep_site_col = FALSE)
      keep_site = pv$keep_site_col %||% FALSE
      all_site_cols = unique(unlist(lapply(per_block, `[[`, "site_cols"), use.names = FALSE))
      all_site_cols = intersect(all_site_cols, names(dt_out))
      feat_cols = if (keep_site) task$feature_names else setdiff(task$feature_names, all_site_cols)

      present = names(dt_out)
      new_roles = roles_orig
      new_roles$feature = setdiff(feat_cols, pk_col)
      for (rn in names(new_roles)) new_roles[[rn]] = intersect(new_roles[[rn]], present)
      new_task$col_roles = new_roles

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

      new_task
    },

    # ---- predict ------------------------------------------------------------

    .predict_task = function(task) {
      st = self$state
      if (is.null(st) || is.null(st$blocks) || !length(st$blocks)) {
        return(task$clone())
      }
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "predict"),
        keep.null = TRUE)

      task_copy = task$clone()
      site_cols_needed = unique(unlist(lapply(st$per_block, function(info) {
        if (identical(info$method, "combat")) {
          info$site_var
        } else {
          info$site_cols
        }
      }), use.names = FALSE))
      cols_needed = unique(c(task_copy$feature_names, site_cols_needed))
      dt = task_copy$data(rows = task_copy$row_ids, cols = cols_needed)

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
              context = sprintf("Block '%s' (partial_corr) prediction design", bn)
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

          Xcorr = if (isTRUE(info$revert)) X + GB else X - GB
          if (!isTRUE(info$zero_center)) {
            means = mb_align_named_numeric(
              info$means,
              cols = colnames(Xcorr),
              context = sprintf(
                "PipeOpSiteCorrection fitted means for block '%s'", bn
              )
            )
            Xcorr = sweep(Xcorr, 2, means, "+")
          }
          if (identical(info$design_kind, "categorical") &&
            !identical(unknown_strategy, "baseline") && any(unseen_mask)) {
            Xcorr[unseen_mask, ] = X[unseen_mask, , drop = FALSE]
          }
          dt[, (Xcols) := as.data.table(Xcorr)]

        } else if (identical(info$method, "combat")) {
          if (!requireNamespace("neuroCombat", quietly = TRUE)) stop("ComBat predict requires 'neuroCombat'.")

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
          if (!length(valid)) next

          # use only the single site var for batch at predict time
          batch_chr = as.character(dt[[info$site_var]])
          known_mask = (batch_chr %in% valid) & !is.na(batch_chr)

          if (identical(combat_policy, "noop")) {
            idx = which(known_mask)
            if (!length(idx)) next
            Yk = t(as.matrix(dt[idx, .SD, .SDcols = Xcols]))
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
            dt[idx, (Xcols) := as.data.table(t(Yck))]
          } else {
            baseline = if (!is.null(info$ref_batch) && info$ref_batch %in% valid) info$ref_batch else valid[1]
            bmap = batch_chr
            bmap[!known_mask | is.na(bmap)] = baseline
            Y = t(as.matrix(dt[, .SD, .SDcols = Xcols]))
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

      out_dt = dt
      row_ids = task$row_ids
      pk_col = mb_make_backend_key_name(c(names(out_dt), task$col_info$id), "..row_id_sitecorr")
      out_dt[, (pk_col) := row_ids]

      # --- bring back all non-feature-role columns from the original task
      roles_orig = task$col_roles
      nonfeat_roles = setdiff(names(roles_orig), "feature")
      extra_cols = unique(unlist(roles_orig[nonfeat_roles], use.names = FALSE))
      extra_cols = setdiff(extra_cols, names(dt)) # avoid duplicates

      if (length(extra_cols)) {
        extra_dt = task$data(rows = task$row_ids, cols = extra_cols)
        dt_out = cbind(dt, extra_dt)
      } else {
        dt_out = dt
      }

      new_task = task_copy$clone()
      new_task$backend = mlr3::as_data_backend(dt_out, primary_key = pk_col)

      # --- features (drop referenced columns from features if keep_site_col = FALSE)
      keep_site = pv$keep_site_col %||% FALSE
      all_site_cols = unique(unlist(lapply(st$per_block, `[[`, "site_cols"), use.names = FALSE))
      all_site_cols = intersect(all_site_cols, names(dt_out))
      feat_cols = if (keep_site) task$feature_names else setdiff(task$feature_names, all_site_cols)

      present = names(dt_out)
      new_roles = roles_orig
      new_roles$feature = setdiff(feat_cols, pk_col)
      for (rn in names(new_roles)) new_roles[[rn]] = intersect(new_roles[[rn]], present)
      new_task$col_roles = new_roles

      new_task
    }
  )
)
