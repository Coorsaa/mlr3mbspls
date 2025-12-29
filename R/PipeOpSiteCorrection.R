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
#'   if estimates were trained with `mod`, it uses an internal mean-imputation
#'   of the training covariate effects. Unseen batches can be handled with
#'   `combat_unknown = "noop"` (skip) or `"baseline"` (map to `ref_batch`).
#'
#' **DIR (`"dir"`, via \pkg{fairmodels})**
#' - Applies distribution repair per block with the given `lambda`.
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
#'         regression (`0` => internally `1e-6`).
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
#' # fake columns
#' task$data[, site := sample(LETTERS[1:3], .N, TRUE)]
#' task$data[, Age := rnorm(.N)]
#' task$data[, Sex := sample(c("F", "M"), .N, TRUE)]
#'
#' po_site = PipeOpSiteCorrection$new(param_vals = list(
#'   blocks = list(num = task$feature_names),
#'   site_correction = list(
#'     num = list(site = "site", covariates = c("Age", "Sex")) # <-- ComBat with mod
#'   ),
#'   method = list(num = "combat"),
#'   keep_site_col = FALSE,
#'   combat_unknown = "noop"
#' ))
#' g = po("copy") %>>% po_site
#' g$train(task)
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
        num = task$feature_names[
          vapply(task$feature_names, function(x) is.numeric(dt[[x]]), logical(1))]
        return(list(.all = num))
      }
      dt_names = names(dt)
      esc = function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\1", s)
      expand_cols = function(cols) {
        unique(unlist(lapply(cols, function(cn) {
          if (cn %in% dt_names) cn else grep(paste0("^", esc(cn), "(\\.|$)"), dt_names, value = TRUE)
        })))
      }
      out = lapply(blocks, function(cols) {
        cols = expand_cols(cols)
        cols[vapply(cols, function(x) is.numeric(dt[[x]]), logical(1))]
      })
      Filter(length, out)
    },

    .method_for_block = function(method_map, bn) {
      if (is.null(method_map)) {
        return("partial_corr")
      }
      m = method_map[[bn]]
      if (is.null(m)) "partial_corr" else as.character(m)
    },

    .encode_site = function(site_vec, known_levels, strategy = c("other", "baseline"), ref_level = NULL) {
      strategy = match.arg(strategy)
      s = as.character(site_vec)
      fac = factor(s, levels = known_levels)
      unseen = setdiff(unique(s), known_levels)
      if (length(unseen)) {
        if (strategy == "other") {
          levels(fac) = c(levels(fac), ".other")
          fac[s %in% unseen] = ".other"
        } else {
          baseline = if (is.null(ref_level)) known_levels[1L] else ref_level
          fac[s %in% unseen] = baseline
          fac = factor(fac, levels = known_levels)
        }
      }
      mm = stats::model.matrix(~fac) # intercept + k-1 dummies
      colnames(mm) = gsub("^fac", "", colnames(mm))
      mm
    },

    .combat_valid_batches = function(est) {
      labs = character(0)
      if (!is.null(est$gamma.hat) && !is.null(rownames(est$gamma.hat))) {
        labs = rownames(est$gamma.hat)
      } else if (!is.null(est$gamma.star) && !is.null(rownames(est$gamma.star))) {
        labs = rownames(est$gamma.star)
      } else if (!is.null(est$batch)) labs <- unique(as.character(est$batch))
      unique(as.character(labs))
    },

    # ---- train --------------------------------------------------------------

    .train_task = function(task) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE)

      blocks = private$.validate_blocks(task, pv$blocks)
      if (is.null(pv$site_correction) || !length(pv$site_correction)) {
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

      idx_fit = if (is.null(pv$subgroup)) {
        seq_len(nrow(dt))
      } else {
        if (is.logical(pv$subgroup)) pv$subgroup else seq_len(nrow(dt)) %in% pv$subgroup
      }

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
          } else {
            # backward compat: single string is the site
            xs = as.character(xspec)
            if (!length(xs)) next
            combat_site = xs[1L]
            combat_covs = character(0)
          }
          site_cols = unique(c(combat_site, combat_covs))
        } else {
          site_cols = as.character(xspec)
        }

        if (!length(site_cols)) next
        missing_sites = setdiff(site_cols, names(dt))
        if (length(missing_sites)) {
          stop(sprintf("Block '%s': missing referenced column(s): %s", bn, paste(missing_sites, collapse = ", ")))
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
        X = as.matrix(dt[, .SD, .SDcols = Xcols])

        if (identical(method, "partial_corr")) {

          single_cat = length(site_cols) == 1L && (is.factor(dt[[site_cols]]) || is.character(dt[[site_cols]]))
          if (single_cat) {
            site_vec = dt[[site_cols]]
            site_lvls = levels(factor(site_vec))
            G_all = private$.encode_site(site_vec, site_lvls, strategy = "other")
            G_fit = G_all[idx_fit, , drop = FALSE]
            design_kind = "categorical"
          } else {
            Z_all = dt[, .SD, .SDcols = site_cols]
            for (cc in names(Z_all)) if (is.character(Z_all[[cc]])) Z_all[, (cc) := factor(get(cc))]
            G_all = stats::model.matrix(~., data = Z_all)
            G_fit = G_all[idx_fit, , drop = FALSE]
            site_lvls = NULL
            design_kind = "matrix"
          }
          design_cols = colnames(G_fit)

          lambda = pv$regularization %||% 0
          if (lambda > 0) {
            beta = cpp_lm_coeff_ridge(
              as.matrix(G_fit),
              as.matrix(X[idx_fit, , drop = FALSE]),
              lambda,
              which(colnames(G_fit) %in% c("(Intercept)")) - 1L
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
          if (!isTRUE(pv$zero_center)) Xcorr <- sweep(Xcorr, 2, mu, "+")
          dt[, (Xcols) := as.data.table(Xcorr)]

          per_block[[bn]] = list(
            method      = "partial_corr",
            site_cols   = site_cols,
            design_cols = design_cols,
            beta        = beta,
            means       = mu,
            zero_center = isTRUE(pv$zero_center),
            revert      = isTRUE(pv$revertflag),
            design_kind = design_kind,
            site_lvls   = site_lvls
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
          dat = data.frame(dt[, .SD, .SDcols = Xcols], protected = prot_vec)
          repaired = fairmodels::disparate_impact_remover(
            data                  = dat,
            protected             = prot_vec,
            features_to_transform = Xcols,
            lambda                = pv$lambda %||% 0.5
          )
          dt[, (Xcols) := as.data.table(repaired[, Xcols, drop = FALSE])]

          per_block[[bn]] = list(
            method    = "dir",
            site_cols = site_cols,
            site_lvls = levels(prot_vec),
            lambda    = pv$lambda %||% 0.5
          )
          blocks_eff[[bn]] = Xcols

        } else {
          stop(sprintf("Unknown method '%s' for block '%s'", method, bn))
        }
      }

      out_dt = dt
      row_ids = task$row_ids
      if (!"..row_id" %in% names(out_dt)) out_dt[, "..row_id" := row_ids]

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
      new_task$backend = mlr3::as_data_backend(dt_out, primary_key = "..row_id")

      # --- features (drop referenced columns from features if keep_site_col = FALSE)
      keep_site = pv$keep_site_col %||% FALSE
      all_site_cols = unique(unlist(lapply(per_block, `[[`, "site_cols"), use.names = FALSE))
      all_site_cols = intersect(all_site_cols, names(dt_out))
      feat_cols = if (keep_site) task$feature_names else setdiff(task$feature_names, all_site_cols)

      present = names(dt_out)
      new_roles = roles_orig
      new_roles$feature = setdiff(feat_cols, "..row_id")
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
          if (is.null(v)) v <- self$param_set$get_values(tags = "train")$combat_unknown
          if (is.null(v)) "noop" else v
        },
        as.character(isTRUE(self$state$keep_site_col))
      )

      new_task
    },

    # ---- predict ------------------------------------------------------------

    .predict_task = function(task) {
      st = self$state
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "predict"),
        keep.null = TRUE)

      task_copy = task$clone()
      site_cols_needed = unique(unlist(lapply(st$per_block, `[[`, "site_cols"), use.names = FALSE))
      cols_needed = unique(c(task_copy$feature_names, site_cols_needed))
      dt = task_copy$data(rows = task_copy$row_ids, cols = cols_needed)

      trained_feats = unique(unlist(st$blocks, use.names = FALSE))
      miss = setdiff(trained_feats, names(dt))
      if (length(miss)) dt[, (miss) := 0.0]

      unknown_strategy = pv$unknown_site %||% st$unknown_site %||% "other"
      combat_policy = pv$combat_unknown %||% "noop"

      for (bn in names(st$blocks)) {
        info = st$per_block[[bn]]
        if (is.null(info)) next

        Xcols = st$blocks[[bn]]
        X = as.matrix(dt[, .SD, .SDcols = Xcols])

        if (identical(info$method, "partial_corr")) {
          # rebuild design
          if (identical(info$design_kind, "categorical")) {
            site_vec = dt[[info$site_cols]]
            s = as.character(site_vec)
            unseen_mask = !(s %in% info$site_lvls) | is.na(s)
            G = private$.encode_site(site_vec, info$site_lvls,
              strategy = ifelse(unknown_strategy == "baseline", "baseline", "other"))
            add = setdiff(info$design_cols, colnames(G))
            if (length(add)) G <- cbind(G, matrix(0, nrow(G), length(add), dimnames = list(NULL, add)))
            drop = setdiff(colnames(G), info$design_cols)
            if (length(drop)) G <- G[, setdiff(colnames(G), drop), drop = FALSE]
            G = G[, info$design_cols, drop = FALSE]
            if (any(unseen_mask)) G[unseen_mask, ] <- 0
          } else {
            Z_new = dt[, .SD, .SDcols = info$site_cols]
            for (cc in names(Z_new)) if (is.character(Z_new[[cc]])) Z_new[, (cc) := factor(get(cc))]
            G = stats::model.matrix(~., data = Z_new)
            add = setdiff(info$design_cols, colnames(G))
            if (length(add)) G <- cbind(G, matrix(0, nrow(G), length(add), dimnames = list(NULL, add)))
            drop = setdiff(colnames(G), info$design_cols)
            if (length(drop)) G <- G[, setdiff(colnames(G), drop), drop = FALSE]
            G = G[, info$design_cols, drop = FALSE]
          }

          if (nrow(G) != nrow(X)) {
            warning(sprintf("Block '%s' (partial_corr): cannot align rows (G %dx%d vs X %dx%d); skipping.",
              bn, nrow(G), ncol(G), nrow(X), ncol(X)))
            next
          }

          beta = info$beta
          if (is.null(rownames(beta))) rownames(beta) <- info$design_cols
          miss_r = setdiff(colnames(G), rownames(beta))
          if (length(miss_r)) {
            beta = rbind(beta, matrix(0, nrow = length(miss_r), ncol = ncol(beta),
              dimnames = list(miss_r, colnames(beta))))
          }
          beta = beta[colnames(G), , drop = FALSE]

          miss_c = setdiff(colnames(beta), colnames(X))
          if (length(miss_c)) {
            X = cbind(X, matrix(0, nrow = nrow(X), ncol = length(miss_c),
              dimnames = list(NULL, miss_c)))
          }
          extra_c = setdiff(colnames(X), colnames(beta))
          if (length(extra_c)) X <- X[, setdiff(colnames(X), extra_c), drop = FALSE]
          X = X[, colnames(beta), drop = FALSE]

          GB = G %*% beta
          if (!identical(colnames(GB), colnames(X))) {
            miss_gb = setdiff(colnames(X), colnames(GB))
            if (length(miss_gb)) {
              GB = cbind(GB, matrix(0, nrow = nrow(GB), ncol = length(miss_gb),
                dimnames = list(NULL, miss_gb)))
            }
            extra_gb = setdiff(colnames(GB), colnames(X))
            if (length(extra_gb)) GB <- GB[, setdiff(colnames(GB), extra_gb), drop = FALSE]
            GB = GB[, colnames(X), drop = FALSE]
          }

          Xcorr = if (isTRUE(info$revert)) X + GB else X - GB
          if (!isTRUE(info$zero_center) && !is.null(info$means) && length(info$means) == ncol(Xcorr)) {
            Xcorr = sweep(Xcorr, 2, info$means, "+")
          }
          dt[, (Xcols) := as.data.table(Xcorr)]

        } else if (identical(info$method, "combat")) {
          if (!("neuroCombat" %in% loadedNamespaces())) stop("ComBat predict requires 'neuroCombat'.")
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
            Yck = tryCatch(neuroCombat::neuroCombatFromTraining(
              dat = Yk, batch = bk, estimates = info$estimates
            )$dat.combat, error = function(e) NULL)
            if (!is.null(Yck)) dt[idx, (Xcols) := as.data.table(t(Yck))]
          } else {
            baseline = if (!is.null(info$ref_batch) && info$ref_batch %in% valid) info$ref_batch else valid[1]
            bmap = batch_chr
            bmap[!known_mask | is.na(bmap)] = baseline
            Y = t(as.matrix(dt[, .SD, .SDcols = Xcols]))
            Yc = tryCatch(neuroCombat::neuroCombatFromTraining(
              dat = Y, batch = factor(bmap, levels = valid), estimates = info$estimates
            )$dat.combat, error = function(e) NULL)
            if (!is.null(Yc)) dt[, (Xcols) := as.data.table(t(Yc))]
          }

        } else if (identical(info$method, "dir")) {
          if (!("fairmodels" %in% loadedNamespaces())) stop("DIR predict requires 'fairmodels'.")
          prot_vec = factor(dt[[info$site_cols]], levels = info$site_lvls)
          dat = data.frame(dt[, .SD, .SDcols = Xcols], protected = prot_vec)
          repaired = fairmodels::disparate_impact_remover(
            data                  = dat,
            protected             = prot_vec,
            features_to_transform = Xcols,
            lambda                = info$lambda
          )
          dt[, (Xcols) := as.data.table(repaired[, Xcols, drop = FALSE])]
        }
      }

      out_dt = dt
      row_ids = task$row_ids
      if (!"..row_id" %in% names(out_dt)) out_dt[, "..row_id" := row_ids]

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
      new_task$backend = mlr3::as_data_backend(dt_out, primary_key = "..row_id")

      # --- features (drop referenced columns from features if keep_site_col = FALSE)
      keep_site = pv$keep_site_col %||% FALSE
      all_site_cols = unique(unlist(lapply(st$per_block, `[[`, "site_cols"), use.names = FALSE))
      all_site_cols = intersect(all_site_cols, names(dt_out))
      feat_cols = if (keep_site) task$feature_names else setdiff(task$feature_names, all_site_cols)

      present = names(dt_out)
      new_roles = roles_orig
      new_roles$feature = setdiff(feat_cols, "..row_id")
      for (rn in names(new_roles)) new_roles[[rn]] = intersect(new_roles[[rn]], present)
      new_task$col_roles = new_roles

      new_task
    }
  )
)
