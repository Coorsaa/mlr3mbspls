#' Supervised Multi-Block Sparse PLS (MB-sPLS-XY) Transformer
#'
#' @title Supervised multi-block sPLS treating the target as an extra block
#'
#' @description
#' **PipeOpMBsPLSXY** is a supervised variant of MB-sPLS. During training, the
#' target (\eqn{Y}) is appended as its own block alongside the input blocks
#' \eqn{X_1,\dots,X_B}, so that extracted components reflect associations
#' among the input and target blocks. The block-wise (Gauss-Seidel) PMD weight
#' updates, the deterministic start and the stopping rule are those of
#' [PipeOpMBsPLS]; correlation criteria control evaluation only. For downstream
#' learners, only the **X-side** latent scores (LVs) are output, at train and
#' at predict time; target-side scores never become features.
#'
#' **Handling encoded column names.** If upstream encoding expands/renames factors
#' into dummy columns like `"base.level"` (e.g., via
#' \code{PipeOpEncode(method = "treatment" | "one-hot")}), you may keep using the
#' **base names** in `blocks` (e.g., `"MINI_dx"`). At `$train()`, base names are
#' resolved to the actual post-encoding columns by [mb_resolve_block_columns()];
#' the resolved blocks must be disjoint. The resolved names are stored and
#' reused at prediction; missing trained columns raise an explicit error
#' instead of being synthesized as zeros.
#'
#' **Centring.** Every retained X column is centred by its training mean; the
#' means are stored in the state and subtracted from prediction data. The target
#' block is centred (and optionally scaled) with its training statistics during
#' training only.
#'
#' For classification tasks, Y is internally one-hot encoded (no intercept).
#' `y_rep` repeats target columns within the same block. This can change the
#' sparsity geometry, but is not an explicit target-block weight because score
#' contributions are standardized. This implementation deflates every block
#' symmetrically, so
#' the requested component count cannot exceed the effective rank of the
#' preprocessed target block. In particular, a centred binary or univariate
#' outcome supports one supervised component. Replicating target columns does
#' not increase their rank.
#'
#' **Sparsity constraints.** Either provide a full \code{c_matrix} (rows = blocks
#' including target, columns = components), or use per-block \code{c_<block>}
#' parameters plus \code{c_target} for the target block.
#'
#' @section State after training:
#' \describe{
#'   \item{\code{blocks_x}}{Resolved X-blocks (numeric, non-constant features).}
#'   \item{\code{center}}{Named list (by X-block) of training column means,
#'         each a numeric vector named by column.}
#'   \item{\code{target_columns}}{Names of the fitted target-block columns.}
#'   \item{\code{ncomp}}{Number of extracted components.}
#'   \item{\code{weights_x}, \code{loadings_x}}{Lists per component with named
#'         weights/loadings per X-block.}
#'   \item{\code{weights_y}, \code{loadings_y}}{Lists per component with the
#'         target-block weights/loadings.}
#'   \item{\code{c_matrix}}{Sparsity matrix actually used when \code{c_matrix}
#'         was supplied (retained X-blocks plus \code{.target}), otherwise
#'         \code{NULL}.}
#'   \item{\code{obj_vec}}{Training objective (MAC/Frobenius) per component,
#'         computed over the X-blocks and the target block.}
#'   \item{\code{p_values}}{Conditional component-wise permutation p-values of
#'         the training diagnostic (\code{NA} when \code{permutation_test =
#'         FALSE}). They assume the supplied preprocessed data and
#'         hyperparameters are fixed and are not full-pipeline inference.}
#'   \item{\code{p_value_scope}}{Description of that scope when the diagnostic
#'         ran, otherwise \code{NULL}.}
#'   \item{\code{converged}, \code{iterations}}{Per component: whether the
#'         solver converged and the number of sweeps it used.}
#'   \item{\code{performance_metric}, \code{correlation_method}}{Settings used
#'         for the objective.}
#'   \item{\code{emit_y_scores}, \code{scores_y}}{Whether target-side scores
#'         were requested and, if so, the training target scores (matrix with
#'         columns \code{LVk_.Y}, rows in training-task order).}
#' }
#'
#' @section Prediction:
#' During `$predict()`, X-scores are computed component-wise with deflation and
#' returned as new columns \code{LVk_<block>}. The Y-block is not needed. All
#' trained X-columns must still be present at prediction time; otherwise the
#' operator errors explicitly.
#'
#' @section Parameters:
#' Hyperparameters are defined in the object's \code{param_set} and can be set
#' via \code{param_vals}.
#' * `blocks` (`uty`, default: the constructor `blocks`; tag `"train"`): named
#'   list mapping X-block names to declared feature names (base names are
#'   resolved to post-encoding columns). Non-numeric columns and columns
#'   without finite positive variance in the training data are dropped, and
#'   blocks without usable columns are dropped.
#' * `ncomp` (`int`, default `1`; tag `"train"`): number of components; at most
#'   the effective rank of the preprocessed target block. Replaced by
#'   `ncol(c_matrix)` when `c_matrix` is set.
#' * `correlation_method` (`fct`, `"pearson"` (default) or `"spearman"`; tags
#'   `c("train", "predict")`): correlation of block scores used for the
#'   objective and the permutation diagnostic.
#' * `performance_metric` (`fct`, `"mac"` (default) or `"frobenius"`; tags
#'   `c("train", "predict")`): latent-correlation summary reported per
#'   component.
#' * `permutation_test` (`lgl`, default `FALSE`; tag `"train"`): run a
#'   conditional component-wise permutation diagnostic after each component and
#'   stop extraction when its p-value exceeds `perm_alpha` (LV1 is always
#'   retained). The p-values are stored in `$state$p_values`. It conditions on
#'   the supplied preprocessed data and hyperparameters and is not a
#'   full-pipeline test.
#' * `n_perm` (`int`, default `100`; tag `"train"`): permutations of the
#'   diagnostic.
#' * `perm_alpha` (`dbl` in `[0, 1]`, default `0.05`; tag `"train"`): cutoff for
#'   the conditional diagnostic.
#' * `c_matrix` (`uty`, default `NULL`; tags `c("train", "tune")`): matrix of L1
#'   budgets (rows = X-blocks and optionally `".target"`, columns = components).
#'   Named rows must be unique and non-empty, and the X rows must match either
#'   all declared or all retained X-blocks. An unnamed matrix is matched by
#'   position to the declared layout: one row per declared X-block, optionally
#'   followed by the target row. A missing target row is filled with
#'   `min(c_target, sqrt(p_target))`. X entries must lie in `[1, sqrt(p_b)]`,
#'   where `p_b` is the structural width of the block (its resolved numeric
#'   columns before constant columns are removed); entries above `sqrt(p)` of
#'   the retained columns are nonbinding, capped there and logged. The target
#'   entry must lie in `[1, sqrt(p_target)]` of the fitted target columns.
#' * `y_rep` (`int`, default `1`; tags `c("train", "tune")`): replications of the
#'   target columns within the target block.
#' * `emit_y_scores` (`lgl`, default `FALSE`; tag `"train"`): store the training
#'   target-block latent scores in `$state$scores_y` for inspection. They are
#'   never added to the task features, because they are functions of the
#'   outcome.
#' * `center_y` (`lgl`, default `TRUE`; tag `"train"`): retained for backward
#'   compatibility. The target block is always centred by its training means;
#'   `FALSE` is ignored with a warning.
#' * `scale_y` (`lgl`, default `TRUE`; tag `"train"`): scale the target columns
#'   to unit training standard deviation.
#' * `c_<block>` (one `dbl` per X-block, lower `1`, upper `sqrt(p_b)` of the
#'   declared names, default `max(1, sqrt(p_b) / 3)`; tags `c("train", "tune")`):
#'   L1 budget of the unit-L2 weight vector of that block.
#' * `c_target` (`dbl` in `[1, 20]`, default `5`; tags `c("train", "tune")`): L1
#'   budget of the target block, capped at `sqrt(p_target)`.
#' * `log_env` (`environment` or `NULL`, default `NULL`; tag `"predict"`): if
#'   set, `$predict()` writes a compact payload (test scores, block names,
#'   `ncomp`, metric) to `log_env$last`.
#'
#' @return A \code{PipeOpMBsPLSXY} that appends columns \code{LVk_<block>} for
#'   each X-block.
#'
#' @seealso
#'   [PipeOpMBsPLS] for the unsupervised variant,
#'   [mlr3pipelines::PipeOpEncode] for post-encoding column names.
#'
#' @keywords internal
#' @importFrom R6 R6Class
#' @import data.table
#' @importFrom checkmate assert_list
#' @importFrom paradox ps p_int p_lgl p_uty p_dbl p_fct
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @export
PipeOpMBsPLSXY = R6::R6Class(
  "PipeOpMBsPLSXY",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @field blocks (`list`) Named base names of X-blocks (resolved post-encoding during training).
    blocks = NULL,

    #' @description Creates a new PipeOpMBsPLSXY instance.
    #' @param id `character(1)` Identifier (default `"mbsplsxy"`).
    #' @param blocks Named list mapping block names to base feature names (required).
    #' @param param_vals Initial `param_set` values.
    #' @return A new PipeOpMBsPLSXY.
    initialize = function(id = "mbsplsxy",
      blocks,
      param_vals = list()) {

      blocks = mb_normalize_blocks(blocks, .var.name = "blocks")

      base_params = list(
        blocks               = paradox::p_uty(tags = "train", default = blocks),
        ncomp                = paradox::p_int(lower = 1L, default = 1L, tags = "train"),
        correlation_method   = paradox::p_fct(c("pearson", "spearman"), default = "pearson", tags = c("train", "predict")),
        performance_metric   = paradox::p_fct(c("mac", "frobenius"), default = "mac", tags = c("train", "predict")),
        permutation_test     = paradox::p_lgl(default = FALSE, tags = "train"),
        n_perm               = paradox::p_int(lower = 1L, default = 100L, tags = "train"),
        perm_alpha           = paradox::p_dbl(lower = 0, upper = 1, default = 0.05, tags = "train"),
        c_matrix             = paradox::p_uty(tags = c("train", "tune"), default = NULL),
        y_rep                = paradox::p_int(lower = 1L, default = 1L, tags = c("train", "tune")),
        emit_y_scores        = paradox::p_lgl(default = FALSE, tags = "train"),
        center_y             = paradox::p_lgl(default = TRUE, tags = "train"),
        scale_y              = paradox::p_lgl(default = TRUE, tags = "train"),
        log_env              = paradox::p_uty(tags = c("predict"), default = NULL)
      )

      for (bn in names(blocks)) {
        p = length(blocks[[bn]])
        base_params[[paste0("c_", bn)]] = paradox::p_dbl(
          lower = 1,
          upper = sqrt(p),
          default = max(1, sqrt(p) / 3),
          tags = c("train", "tune")
        )
      }
      base_params[["c_target"]] = paradox::p_dbl(lower = 1, upper = 20, default = 5, tags = c("train", "tune"))

      if (!is.null(param_vals$c_matrix)) {
        cm = param_vals$c_matrix
        if (!is.matrix(cm) || !is.numeric(cm) || !ncol(cm) ||
          anyNA(cm) || any(!is.finite(cm))) {
          stop(
            "`c_matrix` must be a finite numeric matrix with at least one column.",
            call. = FALSE
          )
        }
        if (!nrow(cm) %in% c(length(blocks), length(blocks) + 1L)) {
          stop(sprintf(
            paste0(
              "c_matrix must have %d rows (X blocks) or %d rows ",
              "(X blocks + '.target'); got %d"
            ),
            length(blocks),
            length(blocks) + 1L,
            nrow(cm)
          ), call. = FALSE)
        }

        if (!is.null(rownames(cm))) {
          if (anyNA(rownames(cm)) || any(!nzchar(rownames(cm))) ||
            anyDuplicated(rownames(cm))) {
            stop("c_matrix row names must be unique and non-empty.",
              call. = FALSE)
          }
          expected_rows = c(
            names(blocks),
            if (".target" %in% rownames(cm)) ".target"
          )
          missing_rows = setdiff(expected_rows, rownames(cm))
          extra_rows = setdiff(rownames(cm), expected_rows)
          if (length(missing_rows) || length(extra_rows)) {
            stop(sprintf(
              paste0(
                "c_matrix rows must match all X blocks and, optionally, ",
                "'.target' exactly. Missing: %s; unexpected: %s"
              ),
              paste(missing_rows, collapse = ", "),
              paste(extra_rows, collapse = ", ")
            ), call. = FALSE)
          }
          cm = cm[expected_rows, , drop = FALSE]
        }
        if (!is.null(colnames(cm)) &&
          (anyNA(colnames(cm)) || any(!nzchar(colnames(cm))) ||
            anyDuplicated(colnames(cm)))) {
          stop("c_matrix column names must be unique and non-empty.",
            call. = FALSE)
        }

        param_vals$c_matrix = cm
        param_vals$ncomp = ncol(cm)
      }

      super$initialize(
        id         = id,
        param_set  = do.call(paradox::ps, base_params),
        param_vals = param_vals
      )

      self$packages = "mlr3mbspls"
      self$blocks = blocks
    }
  ),

  private = list(
    # Safely retrieves the Task for supervised context
    .get_task_safe = function() {
      task = tryCatch(self$input$train_task(), error = function(e) NULL)
      if (!is.null(task)) {
        return(task)
      }
      tryCatch(self$input$truth()$context$task, error = function(e) NULL)
    },

    # Builds the response matrix Y (one-hot for classification, numeric for regression),
    # with optional centering/scaling
    .build_y_matrix = function(task, target_vec, levs, center, scale) {
      if (!is.null(task)) {
        tn = task$target_names
        if (length(tn) != 1L) stop("PipeOpMBsPLSXY: task must have exactly one target.")
        y_vec = task$data(cols = tn)[[1]]

        if (inherits(task, "TaskClassif")) {
          cls = task$class_names[!duplicated(task$class_names)]
          fac = factor(y_vec, levels = cls)
          mm = stats::model.matrix(~ 0 + ., data = data.frame(. = fac))
          colnames(mm) = paste0(".Y_", make.names(colnames(mm), unique = TRUE))
          y_mat = mm

        } else if (inherits(task, "TaskRegr")) {
          y_mat = matrix(as.numeric(y_vec), ncol = 1)
          colnames(y_mat) = ".Y"
        } else {
          stop("PipeOpMBsPLSXY: supports only TaskClassif or TaskRegr.")
        }

      } else {
        if (is.null(target_vec)) stop("PipeOpMBsPLSXY: target missing.")
        if (is.null(levs)) {
          y_mat = matrix(as.numeric(target_vec), ncol = 1)
          colnames(y_mat) = ".Y"
        } else {
          fac = factor(target_vec, levels = levs[!duplicated(levs)])
          mm = stats::model.matrix(~ 0 + ., data = data.frame(. = fac))
          colnames(mm) = paste0(".Y_", make.names(colnames(mm), unique = TRUE))
          y_mat = mm
        }
      }

      y_mat = as.matrix(y_mat)
      storage.mode(y_mat) = "double"

      if (isTRUE(center) || isTRUE(scale)) {
        center_vec = if (isTRUE(center)) colMeans(y_mat, na.rm = TRUE) else rep(0, ncol(y_mat))
        y_mat = sweep(y_mat, 2L, center_vec, FUN = "-")

        if (isTRUE(scale)) {
          scale_vec = apply(y_mat, 2L, stats::sd, na.rm = TRUE)
          scale_vec[!is.finite(scale_vec) | scale_vec < 1e-12] = 1
          y_mat = sweep(y_mat, 2L, scale_vec, FUN = "/")
        }
      }

      as.matrix(y_mat)
    },

    # Convert block columns to matrices
    .as_block_mats = function(dt, blocks) {
      lapply(blocks, function(cols) {
        mat = as.matrix(dt[, ..cols])
        storage.mode(mat) = "double"
        mat
      })
    },

    # Training logic
    .train_dt = function(dt, levels, target = NULL) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE)

      if (!isTRUE(pv$center_y)) {
        warning(
          paste0(
            "PipeOpMBsPLSXY: `center_y = FALSE` is ignored. The target block is ",
            "always centred by its training means because the solver requires ",
            "column-centred blocks."
          ),
          call. = FALSE
        )
      }
      task = private$.get_task_safe()
      y_mat = private$.build_y_matrix(task, target_vec = target, levs = base::levels(target),
        center = TRUE, scale = pv$scale_y)

      y_fit = as.matrix(y_mat)
      storage.mode(y_fit) = "double"
      keep_var = apply(y_fit, 2L, stats::sd, na.rm = TRUE) > 1e-12
      if (!all(keep_var)) {
        y_fit = y_fit[, keep_var, drop = FALSE]
      }
      if (!ncol(y_fit)) {
        stop("PipeOpMBsPLSXY: target matrix has zero variance after preprocessing; cannot fit MB-sPLS-XY.", call. = FALSE)
      }

      rank_y = qr(y_fit)$rank
      requested_components = if (is.matrix(pv$c_matrix)) {
        ncol(pv$c_matrix)
      } else {
        as.integer(pv$ncomp)
      }
      if (requested_components > rank_y) {
        stop(sprintf(
          paste0(
            "PipeOpMBsPLSXY: requested %d components but the preprocessed ",
            "target block has effective rank %d. This symmetric-deflation ",
            "implementation cannot extract more target-associated components ",
            "than that rank; `y_rep` does not increase it."
          ),
          requested_components,
          rank_y
        ), call. = FALSE)
      }

      if (pv$y_rep > 1L) {
        base_names = colnames(y_fit)
        y_fit = do.call(cbind, replicate(pv$y_rep, y_fit, simplify = FALSE))
        rep_id = rep(seq_len(pv$y_rep), each = length(base_names))
        colnames(y_fit) = paste0(rep(base_names, pv$y_rep), "_rep", rep_id)
      }

      resolved = .mb_training_blocks(dt, pv$blocks)
      blocks = resolved$blocks
      if (!length(blocks)) stop("PipeOpMBsPLSXY: no valid X blocks found.")
      X_raw = private$.as_block_mats(dt, blocks)
      names(X_raw) = names(blocks)
      X_raw = lapply(names(X_raw), function(name) {
        .mb_numeric_matrix(
          X_raw[[name]],
          sprintf("training block '%s'", name)
        )
      }) |>
        stats::setNames(names(blocks))
      # The solver assumes column-centred blocks for deflation, EV and scores.
      center = .mb_block_means(X_raw)
      X_list = .mb_center_blocks(X_raw, center)
      .mb_assert_component_rank(
        X_list,
        requested_components,
        "PipeOpMBsPLSXY"
      )

      # Verify row counts are consistent between X blocks and Y matrix
      n_rows_x = nrow(dt)
      n_rows_y = nrow(y_fit)
      if (n_rows_x != n_rows_y) {
        stop(sprintf(
          "PipeOpMBsPLSXY: X blocks have %d rows but the target matrix has %d rows. Ensure the task data and target are aligned.",
          n_rows_x, n_rows_y
        ), call. = FALSE)
      }

      X_list_all = c(X_list, list(.target = as.matrix(y_fit)))
      use_frob = identical(pv$performance_metric, "frobenius")
      # Solver settings of PipeOpMBsPLS: at most 600 Gauss-Seidel sweeps, stop
      # when no block weight changes by 1e-4 or more between sweeps.
      max_iter = 600L
      tol = 1e-4
      cm = NULL

      if (!is.null(pv$c_matrix)) {
        cm = .mb_align_c_matrix(
          pv$c_matrix,
          declared = names(pv$blocks),
          retained = names(blocks),
          extra = ".target"
        )
        if (!".target" %in% rownames(cm)) {
          cm = rbind(cm, matrix(
            min(pv$c_target, sqrt(ncol(y_fit))),
            nrow = 1L,
            ncol = ncol(cm),
            dimnames = list(".target", colnames(cm))
          ))
        }
        # The target block has no data-dependent column filtering: its
        # structural width is its fitted width.
        cm = .mb_prepare_c_matrix(
          blocks = X_list_all,
          c_matrix = cm,
          ncomp = ncol(cm),
          ncomp_missing = FALSE,
          upper_p = c(resolved$p_struct, .target = ncol(y_fit))
        )
        cm = .mb_finalize_c_matrix(cm, X_list_all, self$id)

        fit = cpp_mbspls_multi_lv_cmatrix(
          X_blocks = X_list_all,
          c_matrix = cm,
          max_iter = max_iter,
          tol = tol,
          spearman = identical(pv$correlation_method, "spearman"),
          do_perm = isTRUE(pv$permutation_test),
          n_perm = pv$n_perm,
          alpha = pv$perm_alpha,
          frobenius = use_frob
        )
      } else {
        c_vec = vapply(names(blocks), function(bn) pv[[paste0("c_", bn)]], numeric(1))
        c_vec = c(c_vec, min(pv$c_target, sqrt(ncol(y_fit))))
        names(c_vec)[length(c_vec)] = ".target"
        fit = cpp_mbspls_multi_lv(
          X_blocks = X_list_all, c_constraints = c_vec,
          K = pv$ncomp, max_iter = max_iter, tol = tol,
          spearman = identical(pv$correlation_method, "spearman"),
          do_perm = isTRUE(pv$permutation_test),
          n_perm = pv$n_perm, alpha = pv$perm_alpha,
          frobenius = use_frob
        )
      }

      Bx = length(blocks)
      K = length(fit$W)
      if (K < 1L) stop("PipeOpMBsPLSXY: no components extracted.")
      comp_names = sprintf("LC_%02d", seq_len(K))

      converged = stats::setNames(as.logical(fit$converged %||% rep(NA, K)), comp_names)
      iterations = stats::setNames(
        as.integer(fit$iterations %||% rep(NA_integer_, K)),
        comp_names
      )
      .mb_warn_nonconverged(converged, sprintf("[%s] MB-sPLS-XY", self$id), max_iter)
      p_values = stats::setNames(
        as.numeric(fit$p_values %||% rep(NA_real_, K)),
        comp_names
      )
      if (isTRUE(pv$permutation_test)) {
        lgr$info("[%s] Permutation p-values: %s", self$id, paste(signif(p_values, 3), collapse = ", "))
      }

      y_cols = colnames(y_fit) %||% paste0(".Y_", seq_len(ncol(y_fit)))
      # Loadings of blocks with degenerate scores come back empty; they are zero.
      pad_and_name = function(x, feat_names) {
        x = as.numeric(x)
        if (!length(x)) {
          x = numeric(length(feat_names))
        }
        if (length(x) != length(feat_names)) {
          stop(sprintf("Internal size mismatch: expected %d, got %d", length(feat_names), length(x)))
        }
        stats::setNames(x, feat_names)
      }
      W_X = P_X = vector("list", K)
      W_Y = P_Y = vector("list", K)
      for (k in seq_len(K)) {
        W_X[[k]] = lapply(seq_len(Bx), function(b) pad_and_name(fit$W[[k]][[b]], blocks[[b]]))
        P_X[[k]] = lapply(seq_len(Bx), function(b) pad_and_name(fit$P[[k]][[b]], blocks[[b]]))
        W_Y[[k]] = pad_and_name(fit$W[[k]][[Bx + 1L]], y_cols)
        P_Y[[k]] = pad_and_name(fit$P[[k]][[Bx + 1L]], y_cols)
        names(W_X[[k]]) = names(P_X[[k]]) = names(blocks)
      }
      names(W_X) = names(P_X) = names(W_Y) = names(P_Y) = comp_names

      dt_lat = data.table::as.data.table(
        .mb_deflated_scores(X_list, W_X, P_X, names(blocks))$T
      )

      # Target-side scores are kept for inspection only; they are derived from
      # the outcome and are never added to the task features.
      scores_y = NULL
      if (isTRUE(pv$emit_y_scores)) {
        Y_cur = as.matrix(y_fit)
        scores_y = matrix(0, nrow(Y_cur), K, dimnames = list(NULL, paste0("LV", seq_len(K), "_.Y")))
        for (k in seq_len(K)) {
          ty = drop(Y_cur %*% as.numeric(W_Y[[k]]))
          scores_y[, k] = ty
          if (k < K) {
            Y_cur = Y_cur - ty %*% t(as.numeric(P_Y[[k]]))
          }
        }
      }

      self$state$blocks_x = blocks
      self$state$center = center
      self$state$target_columns = y_cols
      self$state$ncomp = K
      self$state$weights_x = W_X
      self$state$loadings_x = P_X
      self$state$weights_y = W_Y
      self$state$loadings_y = P_Y
      self$state$c_matrix = cm
      self$state$obj_vec = stats::setNames(as.numeric(fit$objective), comp_names)
      self$state$p_values = p_values
      self$state$p_value_scope = if (isTRUE(pv$permutation_test)) {
        .mb_train_p_value_scope("mbspls")
      } else {
        NULL
      }
      self$state$converged = converged
      self$state$iterations = iterations
      self$state$performance_metric = pv$performance_metric
      self$state$correlation_method = pv$correlation_method
      self$state$emit_y_scores = isTRUE(pv$emit_y_scores)
      self$state$scores_y = scores_y

      dt_lat
    },

    # Prediction logic
    .predict_dt = function(dt, levels, target = NULL) {
      st = self$state
      blocks = st$blocks_x
      Bx = length(blocks)
      K = st$ncomp
      if (Bx == 0L || K == 0L) {
        return(data.table::data.table())
      }

      mb_assert_columns_present(
        colnames_dt = names(dt),
        required = unlist(blocks),
        context = sprintf("[%s] Prediction task", self$id),
        hint = "Apply the same preprocessing used during training and retain all trained predictor columns before PipeOpMBsPLSXY."
      )

      X_cur = private$.as_block_mats(dt, blocks)
      names(X_cur) = names(blocks)
      # Apply the training centre (absent in states fitted before centring)
      X_cur = .mb_center_blocks(X_cur, st$center,
        context = sprintf("[%s] Training centre", self$id))

      W_X = lapply(seq_len(K), function(k) {
        lapply(seq_len(Bx), function(b) {
          as.numeric(st$weights_x[[k]][[b]])
        }) |> stats::setNames(names(blocks))
      })
      P_X = lapply(seq_len(K), function(k) {
        lapply(seq_len(Bx), function(b) {
          as.numeric(st$loadings_x[[k]][[b]])
        }) |> stats::setNames(names(blocks))
      })
      dt_lat = data.table::as.data.table(
        .mb_deflated_scores(X_cur, W_X, P_X, names(blocks))$T
      )

      log_env = self$param_set$values$log_env
      if (!is.null(log_env) && inherits(log_env, "environment")) {
        log_env$last = list(
          T_mat       = as.matrix(dt_lat),
          blocks      = names(blocks),
          ncomp       = K,
          perf_metric = st$performance_metric,
          time        = Sys.time()
        )
      }
      dt_lat
    },

    .additional_phash_input = function() {
      list(blocks = self$blocks)
    }
  )
)
