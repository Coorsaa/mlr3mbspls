#' Multi-Block Sparse Partial Least Squares (MB-sPLS) Transformer
#'
#' @title PipeOp \code{mbspls}: extract up to \code{ncomp} orthogonal latent
#'   components from multiple data blocks
#'
#' @description
#' \strong{PipeOpMBsPLS} fits \emph{sequential} MB-sPLS models and appends one
#' latent variable (LV) per block and component to the task's backend.
#' After each component the corresponding rank-1 structure is removed
#' (block-wise score deflation in the sense of Westerhuis et al., 2001),
#' ensuring that successive LVs are orthogonal within every block.
#'
#' The training objective is controlled by \code{performance_metric}:
#' \itemize{
#'   \item \code{"mac"}: mean absolute correlation (average of \eqn{|r|}
#'         across all block-score pairs) per component;
#'   \item \code{"frobenius"}: Frobenius norm of the block-score correlation
#'         matrix \eqn{\sqrt{\sum r^2}}.
#' }
#' We refer to this scalar as the \emph{latent correlation}.
#'
#' During prediction, the same criterion (MAC/Frobenius) and explained variances
#' are computed on test data and, if \code{log_env} is provided, a compact
#' payload is written to \code{log_env$last}. Optionally, \emph{prediction-side
#' validation tests} can be requested (permutation or bootstrap; see Parameters).
#'
#' The operator is a pure transformer: it performs no internal resampling,
#' tuning or preprocessing. Hyper-parameters such as the L1 sparsity levels
#' \eqn{c_\mathrm{block}} are tuned externally (e.g., with \pkg{mlr3tuning}).
#'
#' @section State (after training):
#' \describe{
#'   \item{\code{blocks}}{Named list mapping block names to feature column IDs.}
#'   \item{\code{weights}}{List of length \code{ncomp}; block-specific weight vectors \eqn{w_b^{(k)}}.}
#'   \item{\code{loadings}}{List of block loadings \eqn{p_b^{(k)}} used for deflation.}
#'   \item{\code{ncomp}}{Number of components retained.}
#'   \item{\code{obj_vec}}{Objective values (MAC/Frobenius) per component (training).}
#'   \item{\code{latent_cor_train}}{Objective value of the last retained component (training).}
#'   \item{\code{ev_block}}{Training explained variance per block (rows = components, cols = blocks).}
#'   \item{\code{ev_comp}}{Training explained variance per component (summed across blocks).}
#'   \item{\code{p_values}}{Permutation p-values per component if enabled during training.}
#'   \item{\code{performance_metric}}{\code{"mac"} or \code{"frobenius"}.}
#'   \item{\code{c_matrix}}{If provided/derived, the block-by-component sparsity matrix.}
#'   \item{\code{T_mat}}{Training score matrix (per-component deflation applied);
#'         columns ordered \code{LV1_<block1>, ..., LV1_<blockB>, LV2_<block1>, ...}.}
#'   \item{\code{weights_stable}}{Optional stability-filtered weights (from the bootstrap PipeOp).}
#' }
#'
#' @section Prediction-side logging (\code{log_env$last}):
#' A list containing:
#' \itemize{
#'   \item \code{mac_comp}: numeric vector (length \code{ncomp}) with test MAC/Frobenius per component,
#'   \item \code{ev_block}: matrix \code{(ncomp x n_blocks)} with test per-block explained variances,
#'   \item \code{ev_comp}: numeric vector \code{(ncomp)} with test per-component EV (summed across blocks),
#'   \item \code{T_mat}: test scores \code{(n_test x (ncomp * n_blocks))} with the same column order as training,
#'   \item \code{blocks}: character vector with block names,
#'   \item \code{perf_metric}: objective used (\code{"mac"} or \code{"frobenius"}),
#'   \item \code{time}: POSIXct timestamp,
#'   \item \code{val_test_p}: (if \code{val_test = "permutation"} or \code{"bootstrap"}) per-component p-values,
#'   \item \code{val_test_stat}: (if available) observed test statistic per component,
#'   \item \code{val_bootstrap}: (if \code{val_test = "bootstrap"}) data.table with
#'         observed statistic, bootstrap mean/SE, p-value, and CI per component.
#' }
#'
#' @section Parameters:
#' Hyperparameters are defined in the object's \code{param_set} and can be set
#' via \code{param_vals}. Block membership (\code{blocks}) is a constructor
#' argument and stored in the object state.
#'
#' @param blocks \code{list}. **Required.** Named list assigning each block
#'   name to a character vector of feature column names.
#' @param ncomp \code{integer(1)}. Number of latent components to extract
#'   (or columns of \code{c_matrix} if provided). Default \code{1L}.
#' @param efficient \code{logical(1)}. Reserved flag for an alternative C++ routine.
#' @param correlation_method \code{character(1)}. Correlation estimator for block
#'   scores: \code{"pearson"} (default) or \code{"spearman"}.
#' @param performance_metric \code{character(1)}. Objective to maximize:
#'   \code{"mac"} (mean absolute correlation, default) or \code{"frobenius"}.
#' @param permutation_test \code{logical(1)}. If \code{TRUE} perform a permutation
#'   test after each component during \emph{training} and stop when the empirical p-value
#'   exceeds \code{perm_alpha} (LV1 is always retained).
#' @param n_perm \code{integer(1)}. Number of permutations (training).
#' @param perm_alpha \code{numeric(1)}. Significance level for training-time permutation test.
#' @param c_<block> \code{numeric(1)}. One L1 sparsity limit per block; upper bound defaults to \eqn{\sqrt{p_b}}.
#' @param c_matrix \code{matrix}. Optional matrix of L1 limits (rows = blocks, cols = components).
#' @param store_train_blocks \code{logical(1)}. If \code{TRUE} and \code{log_env} is provided,
#'   store preprocessed training block matrices and sparsity settings in \code{log_env$mbspls_state}.
#' @param predict_weights character; one of "auto","raw","stable_ci","stable_frequency".
#'   Controls which weights PipeOpMBsPLS uses at predict/validation time.
#' @param val_test \code{character(1)}. Prediction-side validation: \code{"none"}, \code{"permutation"}, \code{"bootstrap"}.
#' @param val_test_n \code{integer(1)}. Number of permutations / bootstrap replicates for prediction-side validation.
#' @param val_test_alpha \code{numeric(1)}. Early-stop threshold for permutation and CI level for bootstrap validation.
#' @param val_test_permute_all \code{logical(1)}. If \code{TRUE}, permute all blocks; for \eqn{B=2}, \code{FALSE} permutes block 2 only.
#' @param log_env \code{environment} or \code{NULL}. If not \code{NULL}, writes payloads to \code{log_env$last} and saves a training snapshot in \code{log_env$mbspls_state}.
#' @param append \code{logical(1)}. If \code{TRUE}, keep original features and append LV columns
#'   (both in training and prediction). If \code{FALSE} (default), output only LV columns.
#' @param seed_train \code{integer(1)} or \code{NULL}. Optional random seed for training.
#' @param id character(1). Identifier of the resulting object.
#' @param param_vals named list. List of hyperparameter settings, overwriting the hyperparameter settings that would otherwise be set during construction.
#'
#'
#' @section Construction:
#' `PipeOpMBsPLS$new(id = "mbspls", blocks, param_vals = list())`
#'
#' @section Methods:
#' * `$new(id, blocks, param_vals)` : Initialize the PipeOpMBsPLS.
#'
#' @section Fields:
#' * `blocks` : Named list mapping block names to character vectors of feature names. Set during initialization.
#'
#' @param blocks Named list mapping block names to character vectors of feature names. Set during initialization.
#'
#' @return
#' A \code{PipeOpMBsPLS} that outputs either only \code{LVk_<block>} columns
#' (default) or the original features plus appended LV columns (if \code{append=TRUE}).
#'
#' @family PipeOps
#' @keywords internal
#' @importFrom R6 R6Class
#' @import data.table lgr
#' @importFrom checkmate assert_list
#' @importFrom paradox ps p_int p_lgl p_uty p_dbl p_fct
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @export
PipeOpMBsPLS = R6::R6Class(
  "PipeOpMBsPLS",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @field blocks Named list mapping block names to character vectors of feature names.
    blocks = NULL,

    #' @description Initialize the PipeOpMBsPLS.
    #' @param id character(1). Identifier of the resulting object.
    #' @param blocks named list. Map of block names to feature column names.
    #' @param param_vals named list. List of hyperparameter settings.
    initialize = function(id = "mbspls", blocks, param_vals = list()) {

      checkmate::assert_list(blocks, types = "character", min.len = 1L, names = "unique")

      base_params = list(
        blocks = p_uty(tags = "train", default = blocks),
        ncomp = p_int(lower = 1L, default = 1L, tags = "train"),
        c_matrix = p_uty(tags = c("train", "tune"), default = NULL),
        efficient = p_lgl(default = FALSE, tags = "train"),
        correlation_method = p_fct(c("pearson", "spearman"), default = "pearson", tags = c("train", "predict")),
        performance_metric = p_fct(c("mac", "frobenius"), default = "mac", tags = c("train", "predict")),
        permutation_test = p_lgl(default = FALSE, tags = "train"),
        n_perm = p_int(lower = 1L, default = 100L, tags = "train"),
        perm_alpha = p_dbl(lower = 0, upper = 1, default = 0.05, tags = "train"),
        store_train_blocks = p_lgl(default = FALSE, tags = "train"),
        predict_weights = p_fct(c("auto", "raw", "stable_ci", "stable_frequency"), default = "auto", tags = "predict"),
        val_test = p_fct(c("none", "permutation", "bootstrap"), default = "none", tags = "predict"),
        val_test_alpha = p_dbl(lower = 0, upper = 1, default = 0.05, tags = "predict"),
        val_test_n = p_int(lower = 1L, default = 1000L, tags = "predict"),
        val_test_permute_all = p_lgl(default = TRUE, tags = "predict"),
        log_env = p_uty(tags = c("train", "predict"), default = NULL),
        append = p_lgl(default = FALSE, tags = c("train", "predict")),
        seed_train = p_uty(tags = "train", default = NULL)
      )

      for (bn in names(blocks)) {
        p = length(blocks[[bn]])
        base_params[[paste0("c_", bn)]] = p_dbl(
          lower   = 1,
          upper   = sqrt(p),
          default = max(1, sqrt(p) / 3),
          tags    = c("train", "tune")
        )
      }

      if (!is.null(param_vals$c_matrix)) {
        cm = param_vals$c_matrix
        checkmate::assert_matrix(cm, mode = "numeric", any.missing = FALSE)
        if (nrow(cm) != length(blocks)) {
          stop(sprintf("c_matrix must have %d rows (blocks); got %d", length(blocks), nrow(cm)))
        }
        if (!is.null(rownames(cm))) cm <- cm[names(blocks), , drop = FALSE]
        param_vals$c_matrix = cm
        param_vals$ncomp = ncol(cm)
      }

      super$initialize(id = id, param_set = do.call(ps, base_params), param_vals = param_vals)
      self$packages = "mlr3mbspls"
      self$blocks = blocks
    }
  ),

  private = list(

    .expand_block_cols = function(dt_names, cols) {
      esc = function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\1", s)
      unique(unlist(lapply(cols, function(co) {
        if (co %in% dt_names) co else grep(paste0("^", esc(co), "(\\.|$)"), dt_names, value = TRUE)
      })))
    },

    .with_seed_local = function(seed, fn) {
      with_seed_local(seed, fn)
    },

    # ------------------------------- train -----------------------------------
    .train_dt = function(dt, levels, target = NULL) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE)

      use_frob = (pv$performance_metric == "frobenius")
      blocks = pv$blocks

      dt_names = names(dt)
      blocks = lapply(pv$blocks, function(cols) {
        cand = private$.expand_block_cols(dt_names, cols)
        cand = cand[vapply(cand, function(cl) is.numeric(dt[[cl]]), logical(1))]
        if (!length(cand)) {
          return(character(0))
        }
        keep = vapply(cand, function(cl) stats::var(dt[[cl]], na.rm = TRUE) > 0, logical(1))
        cand[keep]
      })
      blocks = Filter(length, blocks)
      if (!length(blocks)) stop("No block contains at least one numeric, non-constant feature.")
      n_block = length(blocks)

      X_list = lapply(blocks, \(cols) {
        m = as.matrix(dt[, ..cols])
        storage.mode(m) = "double"
        m
      })

      if (!is.null(pv$c_matrix)) {
        cm = pv$c_matrix
        checkmate::assert_matrix(cm, mode = "numeric", any.missing = FALSE)
        if (nrow(cm) != n_block) stop(sprintf("c_matrix must have %d rows (blocks); got %d", n_block, nrow(cm)))
        if (!is.null(rownames(cm))) cm <- cm[names(blocks), , drop = FALSE]
        pv$ncomp = ncol(cm)
        c_matrix = cm
        c_vec = NULL
        lgr$info("Fitting MB-sPLS: %d blocks, %d components (c-matrix).", n_block, pv$ncomp)
      } else {
        c_vec = vapply(names(blocks), \(bn) pv[[paste0("c_", bn)]], numeric(1))
        c_matrix = NULL
        lgr$info("Fitting MB-sPLS: %d blocks, %d components; c = %s",
          n_block, pv$ncomp, paste(c_vec, collapse = ", "))
      }

      fit = private$.with_seed_local(pv$seed_train, function() {
        if (is.null(c_matrix)) {
          cpp_mbspls_multi_lv(
            X_blocks      = X_list,
            c_constraints = c_vec,
            K             = pv$ncomp,
            max_iter      = 600L,
            spearman      = (pv$correlation_method == "spearman"),
            do_perm       = isTRUE(pv$permutation_test),
            n_perm        = pv$n_perm,
            alpha         = pv$perm_alpha,
            frobenius     = use_frob
          )
        } else {
          cpp_mbspls_multi_lv_cmatrix(
            X_blocks  = X_list,
            c_matrix  = c_matrix,
            max_iter  = 600L,
            tol       = 1e-4,
            spearman  = (pv$correlation_method == "spearman"),
            do_perm   = isTRUE(pv$permutation_test),
            n_perm    = pv$n_perm,
            alpha     = pv$perm_alpha,
            frobenius = use_frob
          )
        }
      })

      self$state$c_matrix = c_matrix

      if (length(fit$W) == 0) stop("No components extracted - check sparsity settings.")
      lgr$info("C++ returned %d component(s)", length(fit$W))
      lgr$info("Objectives per component: %s", paste(round(fit$objective, 4), collapse = ", "))
      if (!is.null(fit$p_values)) {
        lgr$info("Permutation p-values: %s", paste(signif(fit$p_values, 3), collapse = ", "))
      }

      W_all = fit$W
      P_all = fit$P
      obj = fit$objective
      pvals = fit$p_values
      ev_blk = fit$ev_block
      ev_cmp = fit$ev_comp

      n_kept = length(W_all)
      B = length(blocks)
      block_names = names(blocks)
      comp_names = sprintf("LC_%02d", seq_len(n_kept))

      pad_and_name = function(x, feat_names) {
        if (length(x) == 0L) x <- numeric(length(feat_names))
        if (length(x) != length(feat_names)) {
          stop(sprintf("Internal size mismatch: expected %d, got %d", length(feat_names), length(x)))
        }
        stats::setNames(x, feat_names)
      }
      for (k in seq_len(n_kept)) {
        for (bn in block_names) {
          feats = blocks[[bn]]
          idx_b = match(bn, names(blocks))
          W_all[[k]][[idx_b]] = pad_and_name(W_all[[k]][[idx_b]], feats)
          P_all[[k]][[idx_b]] = pad_and_name(P_all[[k]][[idx_b]], feats)
        }
        names(W_all[[k]]) = names(P_all[[k]]) = block_names
      }
      names(W_all) = names(P_all) = comp_names

      # Compute training scores
      X_cur = X_list
      score_tables = vector("list", n_kept)
      for (k in seq_len(n_kept)) {
        Wk = W_all[[k]]
        Tk = matrix(0, nrow(dt), B)
        bi = 0L
        for (bn in block_names) {
          bi = bi + 1L
          w_b = Wk[[bn]]
          cols = colnames(X_cur[[bn]])
          if (!is.null(names(w_b))) {
            wv = as.numeric(w_b[cols])
            wv[is.na(wv)] = 0
          } else {
            wv = as.numeric(w_b)
          }
          storage.mode(wv) = "double"
          Tk[, bi] = X_cur[[bn]] %*% wv
        }
        score_tables[[k]] = data.table::as.data.table(Tk)
        data.table::setnames(score_tables[[k]], paste0("LV", k, "_", block_names))
        if (k < n_kept) {
          Pk = P_all[[k]]
          bi = 0L
          for (bn in block_names) {
            bi = bi + 1L
            X_cur[[bn]] = X_cur[[bn]] - Tk[, bi] %*% t(Pk[[bn]])
          }
        }
      }
      dt_lat = do.call(cbind, score_tables)
      T_mat_train = as.matrix(dt_lat)

      self$state$blocks = blocks
      self$state$weights = W_all
      self$state$loadings = P_all
      self$state$ncomp = n_kept
      self$state$T_mat = T_mat_train
      self$state$obj_vec = obj
      self$state$p_values = pvals
      self$state$ev_block = ev_blk
      self$state$ev_comp = ev_cmp
      self$state$latent_cor_train = utils::tail(obj, 1)
      self$state$performance_metric = pv$performance_metric
      self$state$correlation_method = pv$correlation_method

      if (!is.null(pv$log_env) && inherits(pv$log_env, "environment")) {
        sparsity = if (is.null(c_matrix)) {
          cvec = vapply(names(blocks), \(bn) pv[[paste0("c_", bn)]], numeric(1))
          list(type = "c_vec", c_vec = stats::setNames(as.numeric(cvec), names(blocks)))
        } else {
          list(type = "c_matrix", c_matrix = c_matrix)
        }

        payload = list(
          blocks       = blocks,
          ncomp        = n_kept,
          weights      = W_all,
          loadings     = P_all,
          T_mat_train  = T_mat_train,
          comp_names   = sprintf("LC_%02d", seq_len(n_kept)),
          block_names  = names(blocks),
          sparsity     = sparsity,
          corr_method  = pv$correlation_method,
          perf_metric  = pv$performance_metric,
          time         = Sys.time()
        )
        if (isTRUE(pv$store_train_blocks)) payload$X_train_blocks <- X_list
        pv$log_env$mbspls_state = payload
      }

      lgr$info("Training done; last latent correlation = %.4f", utils::tail(obj, 1))

      # ---- output: append or replace
      if (isTRUE(pv$append)) {
        # Append LV columns to the original features
        dt_out = cbind(data.table::as.data.table(dt), dt_lat)
        data.table::setDT(dt_out)
        return(dt_out)
      } else {
        return(dt_lat)
      }
    },

    # ------------------------------- predict ---------------------------------
    .predict_dt = function(dt, levels, target = NULL) {
      pv = utils::modifyList(paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "predict"),
        keep.null = TRUE)

      st = self$state
      block_names = names(st$blocks)
      B = length(block_names)

      # Ensure trained columns exist
      missing_cols = setdiff(unlist(st$blocks), names(dt))
      if (length(missing_cols)) {
        lgr$warn("Adding %d feature columns (all-zero) that were present during training", length(missing_cols))
        dt[, (missing_cols) := 0.0]
      }

      # Build X_test
      X_cur = lapply(st$blocks, function(cols) {
        m = as.matrix(dt[, ..cols])
        storage.mode(m) = "double"
        m
      })
      names(X_cur) = block_names

      # Preserve copy for EV/MAC logging
      X_for_ev = lapply(X_cur, identity)

      # ----------------- choose which weights to use -----------------
      used_source = "raw"
      W_active = st$weights
      P_active = st$loadings
      K_active = length(W_active)

      st_env = NULL
      if (!is.null(pv$log_env) && inherits(pv$log_env, "environment")) {
        st_env = pv$log_env$mbspls_state
      }

      use_env_weights = function(ci = FALSE, freq = FALSE) {
        if (is.null(st_env)) {
          return(FALSE)
        }
        if (ci) {
          if (length(st_env$weights_stable_ci)) {
            W_active <<- st_env$weights_stable_ci
            P_active <<- st_env$loadings_stable_ci %||% NULL
            K_active <<- length(W_active)
            used_source <<- "stable_ci"
            return(TRUE)
          }
          if (identical(st_env$selection_method, "ci") && length(st_env$weights_stable)) {
            W_active <<- st_env$weights_stable
            P_active <<- st_env$loadings_stable %||% NULL
            K_active <<- length(W_active)
            used_source <<- "stable_ci"
            return(TRUE)
          }
        }
        if (freq) {
          if (length(st_env$weights_stable_frequency)) {
            W_active <<- st_env$weights_stable_frequency
            P_active <<- st_env$loadings_stable_frequency %||% NULL
            K_active <<- length(W_active)
            used_source <<- "stable_frequency"
            return(TRUE)
          }
          if (identical(st_env$selection_method, "frequency") && length(st_env$weights_stable)) {
            W_active <<- st_env$weights_stable
            P_active <<- st_env$loadings_stable %||% NULL
            K_active <<- length(W_active)
            used_source <<- "stable_frequency"
            return(TRUE)
          }
        }
        FALSE
      }

      pick = pv$predict_weights %||% "auto"
      if (identical(pick, "auto")) {
        if (!is.null(st_env) && length(st_env$weights_stable)) {
          W_active = st_env$weights_stable
          P_active = st_env$loadings_stable %||% NULL
          K_active = length(W_active)
          used_source = paste0("stable_", st_env$selection_method %||% "ci")
        }
      } else if (identical(pick, "stable_ci")) {
        if (!use_env_weights(ci = TRUE, freq = FALSE)) {
          lgr$warn("predict_weights='stable_ci' requested but not available; falling back to raw.")
          used_source = "raw"
        }
      } else if (identical(pick, "stable_frequency")) {
        if (!use_env_weights(ci = FALSE, freq = TRUE)) {
          lgr$warn("predict_weights='stable_frequency' requested but not available; falling back to raw.")
          used_source = "raw"
        }
      } else {
        used_source = "raw"
      }

      # ---- NOW normalize/pad the *final* chosen weights ----
      for (k in seq_len(K_active)) {
        for (bnm in block_names) {
          feats = colnames(X_for_ev[[bnm]])
          wb = W_active[[k]][[bnm]]
          if (is.null(wb)) {
            W_active[[k]][[bnm]] = stats::setNames(numeric(length(feats)), feats)
          } else if (!is.null(names(wb))) {
            tmp = as.numeric(wb[feats])
            tmp[is.na(tmp)] = 0
            W_active[[k]][[bnm]] = stats::setNames(tmp, feats)
          } else {
            if (length(wb) != length(feats)) {
              W_active[[k]][[bnm]] = stats::setNames(numeric(length(feats)), feats)
            } else {
              W_active[[k]][[bnm]] = stats::setNames(as.numeric(wb), feats)
            }
          }
        }
      }

      # Prepare active state for EV/MAC on the sanitized weights
      st_active = st
      st_active$weights = W_active
      st_active$loadings = P_active
      st_active$ncomp = K_active

      # Then compute EV/MAC safely
      test_ev_results = compute_test_ev(
        X_blocks_test      = X_for_ev,
        W_all              = W_active,
        P_all              = P_active,
        deflate            = TRUE,
        performance_metric = pv$performance_metric,
        correlation_method = pv$correlation_method,
        # if stable weights have no loadings, this will default to test_ls internally
        loading_source     = if (!is.null(P_active) && length(P_active)) "train" else "test_ls"
      )

      use_frob = identical(pv$performance_metric, "frobenius")
      use_spear = identical(pv$correlation_method, "spearman")

      val_test = pv$val_test
      val_test_n = pv$val_test_n
      val_test_permute_all = pv$val_test_permute_all

      val_test_p = rep(NA_real_, K_active)
      val_test_stat = rep(NA_real_, K_active)

      score_tables = vector("list", K_active)
      for (k in seq_len(K_active)) {
        Wk = W_active[[k]]
        Tk = matrix(0, nrow(dt), B)
        bi = 0L
        for (bn in block_names) {
          bi = bi + 1L
          w_b = Wk[[bn]]
          if (is.null(w_b)) {
            Tk[, bi] = 0
            next
          }
          cols = colnames(X_cur[[bn]])
          if (!is.null(names(w_b))) {
            wv = as.numeric(w_b[cols])
            wv[is.na(wv)] = 0
          } else {
            wv = as.numeric(w_b)
            if (length(wv) != length(cols)) wv <- numeric(length(cols))
          }
          storage.mode(wv) = "double"
          Tk[, bi] = X_cur[[bn]] %*% wv
        }
        colnames(Tk) = paste0("LV", k, "_", block_names)
        score_tables[[k]] = data.table::as.data.table(Tk)

        # -------- optional prediction-side validation (permutation) --------
        if (val_test == "permutation" && B >= 2L) {
          Xk_list = lapply(X_cur, function(x) {
            storage.mode(x) = "double"
            x
          })
          res = cpp_perm_test_oos(
            X_test = Xk_list,
            W_trained = Wk,
            n_perm = val_test_n,
            spearman = use_spear,
            frobenius = use_frob,
            permute_all_blocks = isTRUE(val_test_permute_all),
            early_stop_threshold = pv$val_test_alpha
          )
          if (is.list(res)) {
            val_test_p[k] = if (is.null(res$p_value)) NA_real_ else as.numeric(res$p_value)
            val_test_stat[k] = if (is.null(res$stat_obs)) NA_real_ else as.numeric(res$stat_obs)
          } else {
            val_test_p[k] = as.numeric(res)
            val_test_stat[k] = NA_real_
          }
          lgr$info("Component %d: prediction-side permutation test p = %s",
            k, if (is.na(val_test_p[k])) "NA" else formatC(val_test_p[k], digits = 3, format = "f"))
        }

        # -------- optional prediction-side validation (bootstrap) --------
        if (val_test == "bootstrap") {
          Xk_list = lapply(X_cur, function(x) {
            storage.mode(x) = "double"
            x
          })
          n_test = nrow(dt)
          if (n_test >= 10) {
            observed_correlation = 0
            Tk_num = as.matrix(score_tables[[k]])
            if (B >= 2 && nrow(Tk_num) >= 3) {
              sc = c()
              for (i in seq_len(B - 1L)) {
                for (j in (i + 1L):B) {
                  if (stats::sd(Tk_num[, i]) > 0 && stats::sd(Tk_num[, j]) > 0) {
                    r = stats::cor(Tk_num[, i], Tk_num[, j], method = pv$correlation_method)
                    if (is.finite(r)) sc <- c(sc, if (use_frob) r^2 else abs(r))
                  }
                }
              }
              observed_correlation = if (length(sc)) {
                if (use_frob) sqrt(sum(sc)) else mean(sc)
              } else {
                0
              }
            }
            boot_cor = numeric(val_test_n)
            for (rep in seq_len(val_test_n)) {
              idx = sample.int(n_test, replace = TRUE)
              Tk_b = matrix(0, length(idx), B)
              bi = 0L
              for (bn in block_names) {
                bi = bi + 1L
                w_b = Wk[[bn]]
                if (is.null(w_b)) next
                cols = colnames(Xk_list[[bn]])
                if (!is.null(names(w_b))) {
                  wv = as.numeric(w_b[cols])
                  wv[is.na(wv)] = 0
                } else {
                  wv = as.numeric(w_b)
                  if (length(wv) != length(cols)) wv <- numeric(length(cols))
                }
                Tk_b[, bi] = Xk_list[[bn]][idx, , drop = FALSE] %*% wv
              }
              if (B >= 2 && nrow(Tk_b) >= 3) {
                sc = c()
                for (i in seq_len(B - 1L)) {
                  for (j in (i + 1L):B) {
                    if (stats::sd(Tk_b[, i]) > 0 && stats::sd(Tk_b[, j]) > 0) {
                      r = stats::cor(Tk_b[, i], Tk_b[, j], method = pv$correlation_method)
                      if (is.finite(r)) sc <- c(sc, if (use_frob) r^2 else abs(r))
                    }
                  }
                }
                boot_cor[rep] = if (length(sc)) {
                  if (use_frob) sqrt(sum(sc)) else mean(sc)
                } else {
                  0
                }
              } else {
                boot_cor[rep] = 0
              }
            }
            boot_mean = mean(boot_cor, na.rm = TRUE)
            boot_se = stats::sd(boot_cor, na.rm = TRUE)
            valid_boot = boot_cor[is.finite(boot_cor)]
            boot_p_value = if (length(valid_boot) > 0) mean(valid_boot <= observed_correlation) else NA_real_
            conf = 1 - (if (is.null(pv$val_test_alpha)) 0.05 else pv$val_test_alpha)
            zval = stats::qnorm(1 - (1 - conf) / 2)
            ci_lower = boot_mean - zval * boot_se
            ci_upper = boot_mean + zval * boot_se
            val_test_p[k] = boot_p_value
            val_test_stat[k] = observed_correlation
            if (k == 1L) {
              val_bootstrap_results = data.table::data.table(
                component = k, observed_correlation = observed_correlation,
                boot_mean = boot_mean, boot_se = boot_se,
                boot_p_value = boot_p_value,
                boot_ci_lower = ci_lower, boot_ci_upper = ci_upper,
                confidence_level = conf, n_boot = val_test_n
              )
            } else {
              val_bootstrap_results = rbind(val_bootstrap_results, data.table::data.table(
                component = k, observed_correlation = observed_correlation,
                boot_mean = boot_mean, boot_se = boot_se,
                boot_p_value = boot_p_value,
                boot_ci_lower = ci_lower, boot_ci_upper = ci_upper,
                confidence_level = conf, n_boot = val_test_n
              ), fill = TRUE)
            }
          } else {
            lgr$warn("Insufficient test samples (%d) for bootstrap validation", n_test)
          }
        }

        # Deflate for next component if loadings are available
        if (k < K_active) {
          Pk = P_active[[k]]
          if (!is.null(Pk)) {
            bi = 0L
            for (bn in block_names) {
              bi = bi + 1L
              pb = Pk[[bn]]
              if (is.null(pb)) next
              X_cur[[bn]] = X_cur[[bn]] - as.matrix(score_tables[[k]][[bi]]) %*% t(as.matrix(pb))
            }
          }
        }
      }

      dt_lat = do.call(cbind, score_tables)
      T_mat_test = as.matrix(dt_lat)

      ev_block_test = as.matrix(test_ev_results$ev_block)
      ev_comp_test = as.numeric(test_ev_results$ev_comp)
      mac_comp_test = as.numeric(test_ev_results$mac_comp)
      comp_names = sprintf("LC_%02d", seq_len(K_active))
      colnames(ev_block_test) = block_names
      rownames(ev_block_test) = comp_names
      names(ev_comp_test) = comp_names
      names(mac_comp_test) = comp_names

      log_env = self$param_set$values$log_env
      if (!is.null(log_env) && inherits(log_env, "environment")) {
        payload = list(
          mac_comp = mac_comp_test,
          ev_block = ev_block_test,
          ev_comp = ev_comp_test,
          ev_block_cum = as.matrix(test_ev_results$ev_block_cum),
          ev_comp_cum = as.numeric(test_ev_results$ev_comp_cum),
          corr_method = pv$correlation_method,
          T_mat = T_mat_test,
          blocks = block_names,
          perf_metric = pv$performance_metric,
          weights_source = used_source, # <- NEW
          time = Sys.time()
        )
        if (exists("val_test_p", inherits = FALSE) && pv$val_test != "none") {
          names(val_test_p) = comp_names
          names(val_test_stat) = comp_names
          payload$val_test_p = val_test_p
          payload$val_test_stat = val_test_stat
          if (exists("val_bootstrap_results", inherits = FALSE)) {
            payload$val_bootstrap = val_bootstrap_results
          } else {
            payload$val_test_params = list(
              n_perm = if (pv$val_test == "permutation") pv$val_test_n else NA_integer_,
              alpha = pv$val_test_alpha,
              permute_all_blocks = if (pv$val_test == "permutation") isTRUE(val_test_permute_all) else NA
            )
          }
        }
        log_env$last = payload
      }

      # Output (append vs replace) as before
      if (isTRUE(pv$append)) {
        dt_out = cbind(data.table::as.data.table(dt), dt_lat)
        data.table::setDT(dt_out)
        return(dt_out)
      } else {
        return(dt_lat)
      }
    },

    .additional_phash_input = function() {
      list(
        blocks     = self$param_set$values$blocks,
        efficient  = self$param_set$values$efficient,
        c_matrix   = self$param_set$values$c_matrix,
        append     = self$param_set$values$append,
        seed_train = self$param_set$values$seed_train
      )
    }
  )
)
