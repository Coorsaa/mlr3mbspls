#' Supervised Multi-Block Sparse PLS (MB-sPLS-XY) Transformer
#'
#' @title Supervised multi-block sPLS treating the target as an extra block
#'
#' @description
#' **PipeOpMBsPLSXY** is a supervised variant of MB-sPLS. During training, the
#' target (\eqn{Y}) is appended as its own block alongside the input blocks
#' \eqn{X_1,\dots,X_B}, so that extracted components maximize correlation
#' between X-blocks and Y. For downstream learners, only the **X-side** latent
#' scores (LVs) are output (no Y leakage).
#'
#' **Handling encoded column names.** If upstream encoding expands/renames factors
#' into dummy columns like `"base.level"` (e.g., via
#' \code{PipeOpEncode(method = "treatment" | "one-hot")}), you may keep using the
#' **base names** in `blocks` (e.g., `"MINI_dx"`). At `$train()`, base names are
#' expanded to the actual post-encoding columns via regex \code{^<base>(\\.|$)}.
#' The resolved names are stored and reused at prediction; missing trained columns
#' now raise an explicit error instead of being synthesized as zeros.
#'
#' For classification tasks, Y is internally one-hot encoded (no intercept).
#' You can replicate the target block via `y_rep` to increase its weight in the
#' objective.
#'
#' **Sparsity constraints.** Either provide a full \code{c_matrix} (rows = blocks
#' including target, columns = components), or use per-block \code{c_<block>}
#' parameters plus \code{c_target} for the target block.
#'
#' @section State after training:
#' \describe{
#'   \item{\code{blocks_x}}{Resolved X-blocks (numeric, non-constant features).}
#'   \item{\code{ncomp}}{Number of extracted components.}
#'   \item{\code{weights_x}, \code{loadings_x}}{Lists per component with weights/loadings per X-block.}
#'   \item{\code{performance_metric}}{\code{"mac"} (mean absolute correlation) or \code{"frobenius"}.}
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
#'
#' @param blocks \code{list}. **Required.** Named list: block name -> character
#'   vector of base feature names (expanded to post-encoding columns at training).
#' @param ncomp \code{integer(1)}. Number of components to extract (default \code{1L}).
#' @param correlation_method \code{character(1)}. Either \code{"pearson"} (default)
#'   or \code{"spearman"}.
#' @param performance_metric \code{character(1)}. Either \code{"mac"} (default, mean absolute correlation)
#'   or \code{"frobenius"}.
#' @param permutation_test \code{logical(1)}. If \code{TRUE}, run a permutation
#'   test after each latent component (default \code{FALSE}).
#' @param n_perm \code{integer(1)}. Number of permutations.
#' @param perm_alpha \code{numeric(1)}. Significance threshold for permutation test.
#' @param c_matrix \code{matrix} or \code{NULL}. L1 constraints, rows = blocks,
#'   columns = components. If the Y-row is missing, it is automatically added
#'   using \code{c_target}.
#' @param y_rep \code{integer(1)}. Replications of the target block (default \code{1L}).
#' @param emit_y_scores \code{logical(1)}. If \code{TRUE}, also outputs Y-block
#'   latent scores during training (columns \code{LVk_.Y*}). For prediction/ML
#'   pipelines this should remain \code{FALSE} to avoid leakage.
#' @param center_y,scale_y \code{logical(1)}. Centering/scaling of the target
#'   block (defaults \code{TRUE}/\code{TRUE}).
#' @param c_<block> \code{numeric(1)}. (Auto-generated) L1 limit per X-block;
#'   upper bound \eqn{\sqrt{p_b}}.
#' @param c_target \code{numeric(1)}. L1 limit for the target block (default \code{5}).
#' @param log_env \code{environment} or \code{NULL}. If non-\code{NULL}, then
#'   `$predict()` writes a compact result payload to \code{$last}.
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
        emit_y_scores        = paradox::p_lgl(default = FALSE, tags = c("train", "predict")),
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
        checkmate::assert_matrix(cm, mode = "numeric", any.missing = FALSE)
        target_c = as.numeric(param_vals$c_target %||% 5)

        if (!is.null(rownames(cm))) {
          missing_rows = setdiff(names(blocks), rownames(cm))
          if (length(missing_rows)) {
            stop(
              sprintf(
                "c_matrix rows must cover all X blocks. Missing: %s",
                paste(missing_rows, collapse = ", ")
              ),
              call. = FALSE
            )
          }

          cm = cm[names(blocks), , drop = FALSE]
          target_row = matrix(
            target_c,
            nrow = 1L,
            ncol = ncol(cm),
            dimnames = list(".target", colnames(cm))
          )

          if (".target" %in% rownames(param_vals$c_matrix)) {
            cm = rbind(cm, param_vals$c_matrix[".target", , drop = FALSE])
          } else {
            cm = rbind(cm, target_row)
          }
        } else if (nrow(cm) == length(blocks)) {
          cm = rbind(cm, rep(target_c, ncol(cm)))
        } else if (nrow(cm) != length(blocks) + 1L) {
          stop(
            sprintf(
              "c_matrix must have %d rows (X blocks) or %d rows (X blocks + '.target'); got %d",
              length(blocks),
              length(blocks) + 1L,
              nrow(cm)
            ),
            call. = FALSE
          )
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

    # Expand base names to actual columns in dt that match regex ^name(\.|$)
    .expand_block_cols = function(dt_names, cols) {
      esc = function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\1", s)
      unique(unlist(lapply(cols, function(co) {
        if (co %in% dt_names) co else grep(paste0("^", esc(co), "(\\.|$)"), dt_names, value = TRUE)
      })))
    },

    # Clean blocks: expand base names, restrict to numeric and non-constant columns
    .clean_blocks = function(dt, blocks) {
      dt_names = names(dt)
      out = lapply(blocks, function(cols) {
        resolved = private$.expand_block_cols(dt_names, cols)
        resolved = resolved[vapply(resolved, function(cl) is.numeric(dt[[cl]]), logical(1))]
        if (!length(resolved)) {
          return(character(0))
        }
        keep = vapply(resolved, function(cl) mb_has_finite_variance(dt[[cl]]), logical(1))
        resolved[keep]
      })
      Filter(length, out)
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

      task = private$.get_task_safe()
      y_mat = private$.build_y_matrix(task, target_vec = target, levs = base::levels(target),
        center = pv$center_y, scale = pv$scale_y)

      y_fit = as.matrix(y_mat)
      storage.mode(y_fit) = "double"
      keep_var = apply(y_fit, 2L, stats::sd, na.rm = TRUE) > 1e-12
      if (!all(keep_var)) {
        y_fit = y_fit[, keep_var, drop = FALSE]
      }
      if (!ncol(y_fit)) {
        stop("PipeOpMBsPLSXY: target matrix has zero variance after preprocessing; cannot fit MB-sPLS-XY.", call. = FALSE)
      }

      # cpp_mbspls_multi_lv currently fails for rank-1 target blocks; add a
      # deterministic auxiliary target direction when needed.
      rank_y = qr(y_fit)$rank
      if (rank_y < 2L) {
        y1 = as.numeric(y_fit[, 1L])
        aux = as.numeric(seq_len(nrow(y_fit)))
        aux = aux - mean(aux)

        denom = sum(y1 * y1)
        if (is.finite(denom) && denom > 1e-12) {
          aux = aux - (sum(aux * y1) / denom) * y1
        }

        aux_sd = stats::sd(aux)
        if (!is.finite(aux_sd) || aux_sd < 1e-12) {
          stop("PipeOpMBsPLSXY: could not construct a stable target block for fitting.", call. = FALSE)
        }
        aux = aux / aux_sd

        y_fit = cbind(y_fit, aux)
        colnames(y_fit)[ncol(y_fit)] = ".Y_aux"
      }

      if (pv$y_rep > 1L) {
        base_names = colnames(y_fit)
        y_fit = do.call(cbind, replicate(pv$y_rep, y_fit, simplify = FALSE))
        rep_id = rep(seq_len(pv$y_rep), each = length(base_names))
        colnames(y_fit) = paste0(rep(base_names, pv$y_rep), "_rep", rep_id)
      }

      blocks = private$.clean_blocks(dt, pv$blocks)
      if (!length(blocks)) stop("PipeOpMBsPLSXY: no valid X blocks found.")
      X_list = private$.as_block_mats(dt, blocks)

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

      if (!is.null(pv$c_matrix)) {
        cm = pv$c_matrix
        checkmate::assert_matrix(cm, mode = "numeric", any.missing = FALSE)

        if (!is.null(rownames(cm))) {
          missing_rows = setdiff(names(blocks), rownames(cm))
          if (length(missing_rows)) {
            stop("c_matrix rows must cover all retained X blocks. Missing: ", paste(missing_rows, collapse = ", "))
          }

          cm = cm[names(blocks), , drop = FALSE]
          if (".target" %in% rownames(pv$c_matrix)) {
            cm = rbind(cm, pv$c_matrix[".target", , drop = FALSE])
          } else {
            cm = rbind(cm, rep(pv$c_target, ncol(cm)))
            rownames(cm)[nrow(cm)] = ".target"
          }
        } else if (nrow(cm) == length(X_list)) {
          cm = rbind(cm, rep(pv$c_target, ncol(cm)))
        } else if (nrow(cm) != length(X_list) + 1L) {
          stop(
            sprintf(
              "c_matrix must have %d rows (X blocks) or %d rows (X blocks + '.target'); got %d",
              length(X_list),
              length(X_list) + 1L,
              nrow(cm)
            ),
            call. = FALSE
          )
        }

        fit = cpp_mbspls_multi_lv_cmatrix(
          X_blocks = X_list_all,
          c_matrix = cm,
          max_iter = 1000L,
          tol = 1e-4,
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
          K = pv$ncomp, max_iter = 1000L, tol = 1e-6,
          spearman = identical(pv$correlation_method, "spearman"),
          do_perm = isTRUE(pv$permutation_test),
          n_perm = pv$n_perm, alpha = pv$perm_alpha,
          frobenius = use_frob
        )
      }

      Bx = length(blocks)
      K = length(fit$W)
      if (K < 1L) stop("PipeOpMBsPLSXY: no components extracted.")

      y_cols = colnames(y_fit) %||% paste0(".Y_", seq_len(ncol(y_fit)))
      W_X = P_X = vector("list", K)
      W_Y = P_Y = vector("list", K)
      for (k in seq_len(K)) {
        W_X[[k]] = fit$W[[k]][seq_len(Bx)]
        P_X[[k]] = fit$P[[k]][seq_len(Bx)]
        W_Y[[k]] = stats::setNames(as.numeric(fit$W[[k]][[Bx + 1L]]), y_cols)
        P_Y[[k]] = stats::setNames(as.numeric(fit$P[[k]][[Bx + 1L]]), y_cols)
        names(W_X[[k]]) = names(P_X[[k]]) = names(blocks)
      }
      names(W_X) = names(P_X) = names(W_Y) = names(P_Y) = sprintf("LC_%02d", seq_len(K))

      X_cur = X_list
      score_tables = vector("list", K)
      for (k in seq_len(K)) {
        Tk = matrix(0, nrow(dt), Bx)
        for (b in seq_len(Bx)) {
          Tk[, b] = X_cur[[b]] %*% as.numeric(W_X[[k]][[b]])
        }
        score_tables[[k]] = data.table::as.data.table(Tk)
        data.table::setnames(score_tables[[k]], paste0("LV", k, "_", names(blocks)))
        if (k < K) {
          for (b in seq_len(Bx)) {
            Pk = as.numeric(P_X[[k]][[b]])
            X_cur[[b]] = X_cur[[b]] - Tk[, b, drop = FALSE] %*% t(Pk)
          }
        }
      }
      dt_lat = do.call(cbind, score_tables)

      if (isTRUE(pv$emit_y_scores)) {
        Y_cur = as.matrix(y_fit)
        score_tables_y = vector("list", K)
        for (k in seq_len(K)) {
          ty = drop(Y_cur %*% as.numeric(W_Y[[k]]))
          score_tables_y[[k]] = data.table::data.table(ty)
          data.table::setnames(score_tables_y[[k]], paste0("LV", k, "_.Y"))
          if (k < K) {
            py = as.numeric(P_Y[[k]])
            Y_cur = Y_cur - ty %*% t(py)
          }
        }
        dt_lat = cbind(dt_lat, do.call(cbind, score_tables_y))
      }

      self$state$blocks_x = blocks
      self$state$target_columns = y_cols
      self$state$ncomp = K
      self$state$weights_x = W_X
      self$state$loadings_x = P_X
      self$state$weights_y = W_Y
      self$state$loadings_y = P_Y
      self$state$performance_metric = pv$performance_metric
      self$state$correlation_method = pv$correlation_method
      self$state$emit_y_scores = isTRUE(pv$emit_y_scores)

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

      X_cur = lapply(blocks, function(cols) {
        mat = as.matrix(dt[, ..cols])
        storage.mode(mat) = "double"
        mat
      })

      score_tables = vector("list", K)
      for (k in seq_len(K)) {
        Tk = matrix(0, nrow(dt), Bx)
        for (b in seq_len(Bx)) {
          Tk[, b] = X_cur[[b]] %*% as.numeric(st$weights_x[[k]][[b]])
        }
        score_tables[[k]] = data.table::as.data.table(Tk)
        data.table::setnames(score_tables[[k]], paste0("LV", k, "_", names(blocks)))
        if (k < K) {
          for (b in seq_len(Bx)) {
            Pk = as.numeric(st$loadings_x[[k]][[b]])
            X_cur[[b]] = X_cur[[b]] - Tk[, b, drop = FALSE] %*% t(Pk)
          }
        }
      }
      dt_lat = do.call(cbind, score_tables)

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
      list(blocks = self$param_set$values$blocks)
    }
  )
)
