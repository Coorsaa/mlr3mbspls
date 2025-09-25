#' Block-wise Scaling PipeOp for mlr3pipelines
#'
#' @title PipeOpBlockScaling
#' @description
#' Scales **multi-block** feature sets before downstream operators (e.g., MB-sPLS).
#' Supports:
#'  - `"unit_ssq"`: divide every block matrix \(X_b\) by its Frobenius norm
#'    (i.e., `sqrt(sum(X_b^2))`) so each block has unit sum-of-squares.
#'  - `"feature_sd"`: divide each feature by its sample **sd** (no centering);
#'    optionally also divide the whole block by `sqrt(p_b)` so blocks with many
#'    features don't dominate.
#'  - `"feature_zscore"`: z-score each feature (center + sd scaling);
#'    optionally also divide the block by `sqrt(p_b)`.
#'
#' The operator learns scaling parameters on the **training task** and applies
#' them to any new data, ensuring no leakage in resampling.
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
#' # blocks <- list(clin = grep('^clin_', names(dt), value = TRUE),
#' #                geno = grep('^geno_', names(dt), value = TRUE))
#' # po_bs <- PipeOpBlockScaling$new(param_vals = list(
#' #   blocks = blocks, method = "unit_ssq"))
#' # graph <- po_bs %>>% po("mbspls", blocks = blocks)
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

    .collect_blocks = function(dt, blocks, verbose = FALSE) {
      if (is.null(blocks)) {
        num_cols = names(dt)[vapply(dt, is.numeric, logical(1))]
        if (verbose) lgr::lgr$info("Auto-detected %d numeric features for .all block", length(num_cols))
        return(list(.all = num_cols))
      }
      out = lapply(blocks, function(cols) {
        cols = intersect(cols, names(dt))
        cols = cols[vapply(cols, function(cl) is.numeric(dt[[cl]]), logical(1))]
        cols = cols[vapply(cols, function(cl) stats::var(dt[[cl]], na.rm = TRUE) > 0, logical(1))]
        cols
      })
      out = Filter(length, out)
      out
    },

    .train_task = function(task) {
      pv = utils::modifyList(paradox::default_values(self$param_set), self$param_set$get_values(tags = "train"), keep.null = TRUE)
      verbose = isTRUE(pv$verbose)

      task_copy = task$clone()
      dt = task_copy$data(rows = task_copy$row_ids, cols = task_copy$feature_names)

      blocks = private$.collect_blocks(dt, pv$blocks, verbose)
      if (!length(blocks)) stop("PipeOpBlockScaling: no numeric, non-constant features found in any block.")

      method = pv$method %||% "unit_ssq"
      eps = pv$eps %||% 1e-8
      div_p = pv$divide_by_sqrt_p %||% TRUE

      scalers = list()

      # apply scaling in-place
      for (bn in names(blocks)) {
        cols = blocks[[bn]]
        if (!length(cols)) next
        X = as.matrix(dt[, ..cols])

        if (method == "none") {
          scalers[[bn]] = list(type = "none")
        } else if (method == "unit_ssq") {
          alpha = sqrt(sum(X * X, na.rm = TRUE))
          if (!is.finite(alpha) || alpha < eps) alpha <- 1.0
          X = X / alpha
          dt[, (cols) := as.data.table(X)]
          scalers[[bn]] = list(type = "unit_ssq", alpha = alpha)
        } else if (method %in% c("feature_sd", "feature_zscore")) {
          mu = if (method == "feature_zscore") colMeans(X, na.rm = TRUE) else rep(0, ncol(X))
          sd = apply(X, 2, stats::sd, na.rm = TRUE)
          sd[!is.finite(sd) | sd < eps] = 1.0
          Xs = sweep(X, 2, mu, "-")
          Xs = sweep(Xs, 2, sd, "/")
          if (div_p) {
            alpha_p = sqrt(ncol(Xs))
            if (alpha_p > 0) Xs <- Xs / alpha_p else alpha_p <- 1.0
          } else {
            alpha_p = 1.0
          }
          dt[, (cols) := as.data.table(Xs)]
          scalers[[bn]] = list(type = method, mean = mu, sd = sd, alpha_p = alpha_p)
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

      # Rebuild task backend preserving roles
      row_ids = task$row_ids
      if (!".row_id" %in% names(dt)) dt[, ".row_id" := row_ids]
      new_task = task_copy$clone()
      new_task$backend = mlr3::as_data_backend(dt, primary_key = ".row_id")
      new_task$col_roles$feature = setdiff(new_task$feature_names, ".row_id")
      new_task
    },

    .predict_task = function(task) {
      st = self$state
      method = st$method
      eps = st$eps
      div_p = st$div_p

      task_copy = task$clone()
      dt = task_copy$data(rows = task_copy$row_ids, cols = task_copy$feature_names)

      # Ensure training-time columns exist
      missing_cols = setdiff(unlist(st$blocks), names(dt))
      if (length(missing_cols)) {
        lgr::lgr$warn("PipeOpBlockScaling: adding %d missing feature columns at predict time (zeros)", length(missing_cols))
        dt[, (missing_cols) := 0.0]
      }

      for (bn in names(st$blocks)) {
        cols = st$blocks[[bn]]
        if (!length(cols)) next
        X = as.matrix(dt[, ..cols])
        sc = st$scalers[[bn]]
        if (is.null(sc) || identical(sc$type, "none")) {
          next
        } else if (identical(sc$type, "unit_ssq")) {
          alpha = sc$alpha %||% 1.0
          if (!is.finite(alpha) || alpha < eps) alpha <- 1.0
          X = X / alpha
          dt[, (cols) := as.data.table(X)]
        } else if (sc$type %in% c("feature_sd", "feature_zscore")) {
          mu = sc$mean %||% rep(0, ncol(X))
          sd = sc$sd %||% rep(1, ncol(X))
          sd[!is.finite(sd) | sd < eps] = 1.0
          Xs = sweep(X, 2, mu, "-")
          Xs = sweep(Xs, 2, sd, "/")
          alpha_p = sc$alpha_p %||% 1.0
          if (!is.finite(alpha_p) || alpha_p <= 0) alpha_p <- 1.0
          Xs = Xs / alpha_p
          dt[, (cols) := as.data.table(Xs)]
        } else {
          stop("Unknown scaler type in state: ", sc$type)
        }
      }

      # Rebuild task
      row_ids = task$row_ids
      if (!".row_id" %in% names(dt)) dt[, ".row_id" := row_ids]
      new_task = task_copy$clone()
      new_task$backend = mlr3::as_data_backend(dt, primary_key = ".row_id")
      new_task$col_roles$feature = setdiff(new_task$feature_names, ".row_id")
      new_task
    }
  )
)
