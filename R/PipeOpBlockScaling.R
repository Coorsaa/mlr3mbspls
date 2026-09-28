#' Block-wise Scaling PipeOp for mlr3pipelines
#'
#' @title PipeOpBlockScaling
#' @description
#' Scales **multi-block** feature sets before downstream operators (e.g., MB-sPLS).
#' Supports:
#'  - `"unit_ssq"`: center each block, then divide by its Frobenius norm
#'    (i.e., `sqrt(sum(X_b^2))`) so each block has unit sum-of-squares.
#'    Centering is controlled by `center` (default `TRUE`); see below.
#'  - `"feature_sd"`: divide each feature by its sample **sd** (no centering);
#'    optionally also divide the whole block by `sqrt(p_b)` so blocks with many
#'    features don't dominate.
#'  - `"feature_zscore"`: z-score each feature (center + sd scaling);
#'
#' @section Centering and choosing a method:
#' `unit_ssq` centers each block before computing its norm (`center = TRUE`).
#' On uncentered data `sum(X^2)` includes the column means, which inflates the
#' explained-variance denominator by one to two orders of magnitude -- a
#' component can report `ev_block` near 0.95 while explaining little beyond the
#' means.
#'
#' Note that `unit_ssq` applies a **single scalar per block**. It balances the
#' relative weight of whole blocks but does nothing about differences *between
#' features within* a block. If features are on heterogeneous scales, the
#' MB-sPLS gradient `X_b' t` is dominated by the highest-variance features
#' irrespective of centering (the weight update is invariant to block column
#' means), and low-variance true signal will not be selected. Use
#' `"feature_zscore"` (or an upstream `po("scale")`) in that case; a warning is
#' emitted when within-block feature SDs span more than a factor of 20.
#'    optionally also divide the block by `sqrt(p_b)`.
#'
#' The operator learns scaling parameters on the **training task** and applies
#' them to any new data, ensuring no leakage in resampling. Scaled features are
#' replaced in place (as in [mlr3pipelines::PipeOpTaskPreproc]), so column
#' information, keys and all column roles of the output task stay consistent;
#' scaled integer features are returned as numeric features.
#'
#' @section Parameters:
#' * `blocks` (`list`): named list mapping block names to feature columns.
#'   If `NULL`, all numeric features form one block `.all`.
#' * `method` (`"unit_ssq"|"feature_sd"|"feature_zscore"|"none"`):
#'   scaling strategy. Default `"unit_ssq"`.
#' * `divide_by_sqrt_p` (`logical`): when using per-feature scaling, additionally
#'   divide each block by `sqrt(p_b)`. Default `TRUE`.
#' * `eps` (`numeric`): lower bound for standard deviations to avoid division by
#'   ~zero. Default `1e-8`.
#' * `verbose` (`logical`): emit lgr messages. Default `FALSE`.
#'
#' @section State:
#' A list with per-block scaling parameters sufficient to transform new data
#' consistently: either block-level scalar(s) or per-feature means/SDs.
#'
#' @examples
#' # blocks = list(clin = grep("^clin_", names(dt), value = TRUE),
#' #               geno = grep("^geno_", names(dt), value = TRUE))
#' # po_bs = PipeOpBlockScaling$new(param_vals = list(
#' #   blocks = blocks, method = "unit_ssq"))
#' # graph = po_bs %>>% po("mbspls", blocks = blocks)
#'
#' @export
PipeOpBlockScaling = R6::R6Class(
  "PipeOpBlockScaling",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @description Create a new PipeOpBlockScaling.
    #' @param id character(1). Identifier (default: "blockscale").
    #' @param param_vals list. Initial ParamSet values (e.g., blocks/method/etc.).
    initialize = function(id = "blockscale", param_vals = list()) {
      ps = paradox::ps(
        blocks = paradox::p_uty(tags = "train", default = NULL),
        method = paradox::p_fct(levels = c("none", "unit_ssq", "feature_sd", "feature_zscore"),
          default = "unit_ssq", tags = c("train", "predict")),
        divide_by_sqrt_p = paradox::p_lgl(default = TRUE, tags = c("train", "predict")),
        center = paradox::p_lgl(default = TRUE, tags = c("train", "predict")),
        eps = paradox::p_dbl(lower = 0, default = 1e-8, tags = c("train", "predict")),
        verbose = paradox::p_lgl(default = FALSE, tags = c("train", "predict"))
      )

      super$initialize(
        id            = id,
        param_set     = ps,
        param_vals    = param_vals,
        feature_types = c("numeric", "integer", "factor", "character")
      )

      self$packages = c("data.table", "mlr3", "mlr3pipelines", "lgr")
    }
  ),

  private = list(

    .collect_blocks = function(dt, blocks, task = NULL, verbose = FALSE) {
      if (is.null(blocks) && !is.null(task)) {
        blocks = mb_task_blocks(task, context = "PipeOpBlockScaling", allow_null = TRUE)
      }
      if (is.null(blocks)) {
        num_cols = names(dt)[vapply(dt, is.numeric, logical(1))]
        if (verbose) lgr::lgr$info("Auto-detected %d numeric features for .all block", length(num_cols))
        return(list(.all = num_cols))
      }
      mb_resolve_blocks(dt, blocks, numeric_only = TRUE, non_constant = FALSE)
    },

    .scaled_columns = function(blocks, scalers) {
      unique(unlist(lapply(names(blocks), function(bn) {
        if (identical(scalers[[bn]]$type, "none")) character(0) else blocks[[bn]]
      }), use.names = FALSE))
    },

    .train_task = function(task) {
      pv = utils::modifyList(paradox::default_values(self$param_set), self$param_set$get_values(tags = "train"), keep.null = TRUE)
      verbose = isTRUE(pv$verbose)

      dt = task$data(rows = task$row_ids, cols = task$feature_names)

      blocks = private$.collect_blocks(dt, pv$blocks, task = task, verbose = verbose)
      blocks = Filter(length, blocks)
      if (!length(blocks)) stop("PipeOpBlockScaling: no numeric features found in any block.")

      method = pv$method %||% "unit_ssq"
      eps = pv$eps %||% 1e-8
      div_p = pv$divide_by_sqrt_p %||% TRUE
      do_center = pv$center %||% TRUE
      if (length(eps) != 1L || !is.numeric(eps) || !is.finite(eps) || eps <= 0) {
        stop("PipeOpBlockScaling: `eps` must be one finite positive number.",
          call. = FALSE)
      }

      scalers = list()

      # apply scaling in-place
      for (bn in names(blocks)) {
        cols = blocks[[bn]]
        if (!length(cols)) next
        X = .mb_numeric_matrix(as.matrix(dt[, ..cols]),
          sprintf("training block '%s'", bn))
        colnames(X) = cols

        if (method == "none") {
          scalers[[bn]] = list(type = "none", columns = cols)
        } else if (method == "unit_ssq") {
          # Centre before taking the block norm. On uncentred data sum(X^2) is a
          # raw sum of squares that includes the column means, which inflates the
          # explained-variance denominator by one to two orders of magnitude: a
          # component can report ev_block ~0.95 while explaining little beyond
          # the means. Block scaling in the RGCCA sense is applied to centred
          # data. Set center = FALSE if the data are already centred.
          mu = if (isTRUE(do_center)) colMeans(X) else rep(0, ncol(X))
          X = sweep(X, 2, mu, "-")
          alpha = sqrt(sum(X * X))
          if (!is.finite(alpha) || alpha <= eps) {
            stop(sprintf(
              "PipeOpBlockScaling: training block '%s' has zero or non-finite sum of squares.",
              bn
            ), call. = FALSE)
          }
          X = X / alpha
          dt[, (cols) := as.data.table(X)]

          # unit_ssq applies a single scalar per block and therefore cannot
          # equalise features *within* a block. Where that clearly matters the
          # MB-sPLS gradient X_b' t is dominated by the highest-variance
          # features, and centring does not help (the update is invariant to
          # block column means). Flag it.
          sds = apply(X, 2, stats::sd)
          sds = sds[is.finite(sds) & sds > 0]
          if (length(sds) > 1L && max(sds) / min(sds) > 20) {
            lgr::lgr$warn(paste0(
              "[%s] block '%s': feature SDs span a factor of %.0f after unit_ssq ",
              "scaling. unit_ssq normalises whole blocks and cannot equalise ",
              "features within them, so weights will be dominated by the ",
              "highest-variance features. Consider method = 'feature_zscore'."),
              self$id, bn, max(sds) / min(sds))
          }
          scalers[[bn]] = list(
            type = "unit_ssq", alpha = alpha, mean = mu, columns = cols
          )
        } else if (method %in% c("feature_sd", "feature_zscore")) {
          mu = if (method == "feature_zscore") colMeans(X) else rep(0, ncol(X))
          sd = apply(X, 2, stats::sd)
          bad = !is.finite(sd) | sd <= eps
          if (any(bad)) {
            stop(sprintf(
              paste0(
                "PipeOpBlockScaling: zero-variance or numerically constant ",
                "training predictors in block '%s': %s."
              ),
              bn, paste(cols[bad], collapse = ", ")
            ), call. = FALSE)
          }
          names(mu) = names(sd) = cols
          Xs = sweep(X, 2, mu, "-")
          Xs = sweep(Xs, 2, sd, "/")
          if (div_p) {
            alpha_p = sqrt(ncol(Xs))
            if (alpha_p > 0) {
              Xs = Xs / alpha_p
            } else {
              alpha_p = 1.0
            }
          } else {
            alpha_p = 1.0
          }
          dt[, (cols) := as.data.table(Xs)]
          scalers[[bn]] = list(
            type = method, mean = mu, sd = sd, alpha_p = alpha_p,
            columns = cols
          )
        } else {
          stop("Unknown method: ", method)
        }
      }

      self$state = list(
        blocks   = blocks,
        method   = method,
        eps      = eps,
        div_p    = div_p,
        scalers  = scalers
      )

      # Replace scaled features in place so that column information, keys and
      # roles stay consistent with the backend.
      mb_task_replace_features(task, dt, changed = private$.scaled_columns(blocks, scalers))
      task
    },

    .predict_task = function(task) {
      st = self$state
      method = st$method
      eps = st$eps
      div_p = st$div_p

      dt = task$data(rows = task$row_ids, cols = task$feature_names)

      # Ensure training-time columns exist
      mb_assert_columns_present(
        colnames_dt = names(dt),
        required = unlist(st$blocks),
        context = sprintf("[%s] Prediction task", self$id),
        hint = "Apply the same preprocessing used during training and retain all block features before PipeOpBlockScaling."
      )

      for (bn in names(st$blocks)) {
        cols = st$blocks[[bn]]
        if (!length(cols)) next
        X = .mb_numeric_matrix(as.matrix(dt[, ..cols]),
          sprintf("prediction block '%s'", bn))
        colnames(X) = cols
        sc = st$scalers[[bn]]
        if (!is.null(sc$columns) && !identical(as.character(sc$columns), cols)) {
          stop(sprintf(
            "PipeOpBlockScaling: stored schema for block '%s' is inconsistent with its training columns.",
            bn
          ), call. = FALSE)
        }
        if (is.null(sc)) {
          stop(sprintf(
            "PipeOpBlockScaling: fitted scaler state is missing for block '%s'.",
            bn
          ), call. = FALSE)
        }
        if (length(sc$type) != 1L || !is.character(sc$type) ||
          is.na(sc$type) || !nzchar(sc$type)) {
          stop(sprintf(
            "PipeOpBlockScaling: invalid fitted scaler type for block '%s'.",
            bn
          ), call. = FALSE)
        }
        if (identical(sc$type, "none")) {
          next
        } else if (identical(sc$type, "unit_ssq")) {
          alpha = sc$alpha
          if (length(alpha) != 1L || !is.numeric(alpha) ||
            !is.finite(alpha) || alpha <= eps) {
            stop(sprintf(
              "PipeOpBlockScaling: invalid fitted unit-SSQ state for block '%s'.",
              bn
            ), call. = FALSE)
          }
          if (!is.null(sc$mean)) {
            mu = mb_align_named_numeric(
              sc$mean,
              cols = cols,
              context = sprintf(
                "PipeOpBlockScaling fitted unit-SSQ means for block '%s'", bn
              )
            )
            X = sweep(X, 2, mu, "-")
          }
          X = X / alpha
          dt[, (cols) := as.data.table(X)]
        } else if (sc$type %in% c("feature_sd", "feature_zscore")) {
          mu = mb_align_named_numeric(
            sc$mean,
            cols = cols,
            context = sprintf(
              "PipeOpBlockScaling fitted means for block '%s'", bn
            )
          )
          sd = mb_align_named_numeric(
            sc$sd,
            cols = cols,
            context = sprintf(
              "PipeOpBlockScaling fitted scales for block '%s'", bn
            )
          )
          if (any(sd <= eps)) {
            stop(sprintf(
              "PipeOpBlockScaling: invalid fitted feature-scaling state for block '%s'.",
              bn
            ), call. = FALSE)
          }
          Xs = sweep(X, 2, mu, "-")
          Xs = sweep(Xs, 2, sd, "/")
          alpha_p = sc$alpha_p
          if (length(alpha_p) != 1L || !is.numeric(alpha_p) ||
            !is.finite(alpha_p) || alpha_p <= 0) {
            stop(sprintf(
              "PipeOpBlockScaling: invalid fitted block-size state for block '%s'.",
              bn
            ), call. = FALSE)
          }
          Xs = Xs / alpha_p
          dt[, (cols) := as.data.table(Xs)]
        } else {
          stop("Unknown scaler type in state: ", sc$type)
        }
      }

      mb_task_replace_features(task, dt, changed = private$.scaled_columns(st$blocks, st$scalers))
      task
    }
  )
)
