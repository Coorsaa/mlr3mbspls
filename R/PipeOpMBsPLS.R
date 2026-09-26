#' Multi-Block Sparse Partial Least Squares (MB-sPLS) Transformer
#'
#' @title Extract up to \code{ncomp} orthogonal latent components from multiple data blocks
#'
#' @description
#' \strong{PipeOpMBsPLS} fits \emph{sequential} MB-sPLS models and appends one
#' latent variable (LV) per block and component to the task's backend.
#' After each component the corresponding rank-1 structure is removed
#' (block-wise score deflation in the sense of Westerhuis et al., 2001),
#' ensuring that successive LVs are orthogonal within every block.
#'
#' The association criterion used for convergence and evaluation is controlled
#' by \code{performance_metric}:
#' \itemize{
#'   \item \code{"mac"}: mean absolute correlation (average of \eqn{|r|}
#'         across all block-score pairs) per component;
#'   \item \code{"frobenius"}: Euclidean norm of the distinct off-diagonal
#'         block-score correlations \eqn{\sqrt{\sum_{i<j} r_{ij}^2}}.
#' }
#' We refer to this scalar as the \emph{latent correlation}. Weights are updated
#' by constrained PMD steps against standardized scores from the other blocks.
#' These updates do not directly optimize the reported correlation criterion;
#' changing the criterion changes convergence, evaluation, and tuning rather
#' than the weight-update formula.
#'
#' During prediction, the same criterion (MAC/Frobenius) and explained variances
#' are computed on test data and, if \code{log_env} is provided, a compact
#' payload is written to \code{log_env$last}. Optionally, prediction-side
#' diagnostics can be requested (permutation or descriptive bootstrap; see
#' Parameters).
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
#'   \item{\code{p_values}}{Conditional component-wise permutation p-values if
#'         enabled during training. They assume the supplied preprocessed data
#'         and hyperparameters are fixed and are not full-pipeline inference.}
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
#'   \item \code{val_test_p}: (if \code{val_test = "permutation"}) per-component
#'         conditional permutation p-values,
#'   \item \code{val_test_stat}: (if available) observed test statistic per component,
#'   \item \code{val_bootstrap}: (if \code{val_test = "bootstrap"}) data.table with
#'         the observed statistic, bias, standard error, interval, confidence
#'         level, and effective replicate count. Its p-value fields are `NA`
#'         because an ordinary bootstrap distribution is not a null distribution.
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
#' @param performance_metric \code{character(1)}. Association criterion for
#'   convergence, tuning, and evaluation:
#'   \code{"mac"} (mean absolute correlation, default) or \code{"frobenius"}.
#' @param permutation_test \code{logical(1)}. If \code{TRUE}, perform a
#'   conditional component-wise permutation diagnostic during training and stop
#'   when its empirical p-value exceeds \code{perm_alpha} (LV1 is always
#'   retained). This does not replace a design-valid full-pipeline permutation
#'   analysis.
#' @param n_perm \code{integer(1)}. Number of permutations (training).
#' @param perm_alpha \code{numeric(1)}. Cutoff for the conditional train-time
#'   permutation diagnostic.
#' @param c_<block> \code{numeric(1)}. One L1 sparsity limit per block; upper bound defaults to \eqn{\sqrt{p_b}}.
#' @param c_matrix \code{matrix}. Optional matrix of L1 limits (rows = blocks, cols = components).
#' @param store_train_blocks \code{logical(1)}. If \code{TRUE} and \code{log_env} is provided,
#'   store preprocessed training block matrices and sparsity settings in \code{log_env$mbspls_state}.
#' @param predict_weights character; one of "auto","raw","stable_ci","stable_frequency".
#'   Controls which weights PipeOpMBsPLS uses at predict/validation time. Explicit
#'   requests for \code{"stable_ci"} or \code{"stable_frequency"} now error if the
#'   requested stability-selected weights/loadings are unavailable in \code{log_env}.
#' @param val_test \code{character(1)}. Prediction-side diagnostic:
#'   \code{"none"}, conditional held-out \code{"permutation"}, or descriptive
#'   \code{"bootstrap"}. A design-level hypothesis test must rerun the complete
#'   preprocessing, tuning, and fitting pipeline within each valid permutation.
#' @param val_test_n \code{integer(1)}. Number of permutations / bootstrap replicates for prediction-side validation.
#' @param val_test_alpha \code{numeric(1)}. Alpha used for the descriptive
#'   bootstrap confidence interval. Retained for permutation calls for API
#'   compatibility; sampled permutation p-values always use all replicates.
#' @param val_test_permute_all \code{logical(1)}. If \code{TRUE}, permute all blocks; for \eqn{B=2}, \code{FALSE} permutes block 2 only.
#' @param seed_validation \code{integer(1)} or \code{NULL}. Optional seed for
#'   prediction-side permutation/bootstrap diagnostics. One L'Ecuyer-CMRG stream
#'   is assigned per component and the caller's RNG state is restored.
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

      blocks = mb_normalize_blocks(blocks, .var.name = "blocks")

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
        seed_validation = p_uty(tags = "predict", default = NULL),
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
        if (!is.matrix(cm) || !is.numeric(cm) || !ncol(cm) ||
          anyNA(cm) || any(!is.finite(cm))) {
          stop(
            "`c_matrix` must be a finite numeric matrix with at least one column.",
            call. = FALSE
          )
        }
        if (nrow(cm) != length(blocks)) {
          stop(sprintf(
            "c_matrix must have %d rows (blocks); got %d",
            length(blocks),
            nrow(cm)
          ), call. = FALSE)
        }
        if (!is.null(rownames(cm))) {
          if (anyNA(rownames(cm)) || any(!nzchar(rownames(cm))) ||
            anyDuplicated(rownames(cm))) {
            stop("c_matrix row names must be unique and non-empty.",
              call. = FALSE)
          }
          missing_rows = setdiff(names(blocks), rownames(cm))
          extra_rows = setdiff(rownames(cm), names(blocks))
          if (length(missing_rows) || length(extra_rows)) {
            stop(sprintf(
              "c_matrix rows must match all blocks exactly. Missing: %s; unexpected: %s",
              paste(missing_rows, collapse = ", "),
              paste(extra_rows, collapse = ", ")
            ), call. = FALSE)
          }
          cm = cm[names(blocks), , drop = FALSE]
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
        keep = vapply(cand, function(cl) mb_has_finite_variance(dt[[cl]]), logical(1))
        cand[keep]
      })
      blocks = Filter(length, blocks)
      if (!length(blocks)) stop("No block contains at least one numeric, non-constant feature.")
      n_block = length(blocks)

      X_list = lapply(names(blocks), function(name) {
        cols = blocks[[name]]
        m = .mb_numeric_matrix(
          as.matrix(dt[, ..cols]),
          sprintf("training block '%s'", name)
        )
        storage.mode(m) = "double"
        m
      }) |>
        stats::setNames(names(blocks))

      if (!is.null(pv$c_matrix)) {
        cm_input = pv$c_matrix
        if (!is.null(rownames(cm_input))) {
          declared_names = names(pv$blocks)
          retained_names = names(blocks)
          valid_row_set = setequal(rownames(cm_input), declared_names) ||
            setequal(rownames(cm_input), retained_names)
          if (!valid_row_set) {
            stop(
              paste0(
                "Named c_matrix rows must match either all declared or all ",
                "retained blocks exactly."
              ),
              call. = FALSE
            )
          }
          cm_input = cm_input[retained_names, , drop = FALSE]
        }
        cm = .mb_prepare_c_matrix(
          blocks = X_list,
          c_matrix = cm_input,
          ncomp = ncol(cm_input),
          ncomp_missing = FALSE
        )
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
      .mb_assert_component_rank(X_list, pv$ncomp, "PipeOpMBsPLS")

      fit = with_seed_local(pv$seed_train, function() {
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
        if (length(x) == 0L) {
          x = numeric(length(feat_names))
        }
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
      self$state$p_value_scope = if (isTRUE(pv$permutation_test)) {
        paste(
          "Conditional component-wise diagnostic with fixed preprocessing and",
          "hyperparameters; not a full-pipeline permutation test."
        )
      } else {
        NULL
      }
      self$state$ev_block = ev_blk
      self$state$ev_comp = ev_cmp
      self$state$latent_cor_train = utils::tail(obj, 1)
      self$state$performance_metric = pv$performance_metric
      self$state$correlation_method = pv$correlation_method
      self$state$pkg_version = as.character(utils::packageVersion("mlr3mbspls"))

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
        if (isTRUE(pv$store_train_blocks)) {
          payload$X_train_blocks = X_list
        }
        payload$pipeop_id = self$id
        run_id = log_env_store_state(pv$log_env, payload, warn_overwrite = TRUE)
        self$state$run_id = run_id
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
      mb_assert_columns_present(
        colnames_dt = names(dt),
        required = unlist(st$blocks),
        context = sprintf("[%s] Prediction task", self$id),
        hint = "Apply the same preprocessing used during training and retain all trained feature columns before PipeOpMBsPLS."
      )

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
      run_id_for_lookup = st$run_id %||% NULL
      if (!is.null(run_id_for_lookup) && nzchar(as.character(run_id_for_lookup)) &&
        !is.null(pv$log_env) && inherits(pv$log_env, "environment")) {
        st_env = tryCatch(
          .mbspls_state_from_env(
            pv$log_env,
            run_id = as.character(run_id_for_lookup),
            require_train_blocks = FALSE,
            where = "log_env"
          ),
          error = function(e) NULL
        )
      }

      get_env_weights = function(ci = FALSE, freq = FALSE) {
        if (is.null(st_env)) {
          return(NULL)
        }
        if (ci) {
          if (length(st_env$weights_stable_ci)) {
            return(list(
              weights = st_env$weights_stable_ci,
              loadings = st_env$loadings_stable_ci %||% NULL,
              source = "stable_ci"
            ))
          }
          if (identical(st_env$selection_method, "ci") && length(st_env$weights_stable)) {
            return(list(
              weights = st_env$weights_stable,
              loadings = st_env$loadings_stable %||% NULL,
              source = "stable_ci"
            ))
          }
        }
        if (freq) {
          if (length(st_env$weights_stable_frequency)) {
            return(list(
              weights = st_env$weights_stable_frequency,
              loadings = st_env$loadings_stable_frequency %||% NULL,
              source = "stable_frequency"
            ))
          }
          if (identical(st_env$selection_method, "frequency") && length(st_env$weights_stable)) {
            return(list(
              weights = st_env$weights_stable,
              loadings = st_env$loadings_stable %||% NULL,
              source = "stable_frequency"
            ))
          }
        }
        NULL
      }

      stable_request_error = function(requested) {
        requested = as.character(requested)[1L]
        run_label = if (!is.null(run_id_for_lookup) && nzchar(as.character(run_id_for_lookup))) {
          as.character(run_id_for_lookup)
        } else {
          "<latest>"
        }
        selection = switch(requested,
          stable_ci = "ci",
          stable_frequency = "frequency",
          requested
        )
        stop(sprintf(
          "predict_weights='%s' was requested, but matching stability-selected weights/loadings are not available in log_env for run_id='%s'. Run PipeOpMBsPLSBootstrapSelect with selection_method='%s' on the same log_env, or use predict_weights='raw'.",
          requested,
          run_label,
          selection
        ), call. = FALSE)
      }

      pick = pv$predict_weights %||% "auto"
      if (identical(pick, "auto")) {
        if (!is.null(st_env) && length(st_env$weights_stable)) {
          W_active = st_env$weights_stable
          P_active = st_env$loadings_stable %||% NULL
          K_active = length(W_active)
          used_source = paste0("stable_", st_env$selection_method %||% "ci")
          # Guard: if all components have empty block weights, fall back to raw
          all_populated = K_active > 0L && all(vapply(seq_len(K_active), function(ki) {
            is.list(W_active[[ki]]) && length(W_active[[ki]]) > 0L &&
              all(vapply(W_active[[ki]], function(w) length(w) > 0L, logical(1L)))
          }, logical(1L)))
          if (all_populated) {
            lgr$info("[%s] predict_weights='auto': using '%s' from log_env.", self$id, used_source)
          } else {
            lgr$warn("[%s] predict_weights='auto': stable weights exist but have 0 components or empty block weights; falling back to 'raw'.", self$id)
            W_active = st$weights
            P_active = st$loadings
            K_active = length(W_active)
            used_source = "raw"
          }
        } else {
          lgr$info("[%s] predict_weights='auto': no stable weights found in log_env; falling back to 'raw'.", self$id)
        }
      } else if (identical(pick, "stable_ci")) {
        selected = get_env_weights(ci = TRUE, freq = FALSE)
        if (is.null(selected)) {
          stable_request_error("stable_ci")
        }
        W_active = selected$weights
        P_active = selected$loadings
        K_active = length(W_active)
        used_source = selected$source
      } else if (identical(pick, "stable_frequency")) {
        selected = get_env_weights(ci = FALSE, freq = TRUE)
        if (is.null(selected)) {
          stable_request_error("stable_frequency")
        }
        W_active = selected$weights
        P_active = selected$loadings
        K_active = length(W_active)
        used_source = selected$source
      } else {
        used_source = "raw"
      }

      if (!is.list(W_active) || length(W_active) < K_active) {
        stop(sprintf(
          "Selected prediction weights are inconsistent: expected %d component(s), got %d.",
          K_active,
          length(W_active %||% list())
        ), call. = FALSE)
      }
      if (!is.list(P_active) || length(P_active) < K_active) {
        stop(sprintf(
          "Selected prediction loadings are unavailable or incomplete for weights_source='%s'. Prediction requires matching loadings to preserve the trained deflation path.",
          used_source
        ), call. = FALSE)
      }

      # ---- align the final chosen weights/loadings strictly to trained features ----
      for (k in seq_len(K_active)) {
        if (!is.list(W_active[[k]])) {
          stop(sprintf("Component %d of the selected prediction weights is not a block-wise list.", k), call. = FALSE)
        }
        if (!is.list(P_active[[k]])) {
          stop(sprintf("Component %d of the selected prediction loadings is not a block-wise list.", k), call. = FALSE)
        }
        for (bnm in block_names) {
          feats = colnames(X_for_ev[[bnm]])
          W_active[[k]][[bnm]] = mb_align_named_numeric(
            W_active[[k]][[bnm]],
            cols = feats,
            context = sprintf("Prediction weights for component %d, block '%s'", k, bnm)
          )
          P_active[[k]][[bnm]] = mb_align_named_numeric(
            P_active[[k]][[bnm]],
            cols = feats,
            context = sprintf("Prediction loadings for component %d, block '%s'", k, bnm)
          )
        }
      }

      # Then compute EV/MAC safely
      test_ev_results = compute_test_ev(
        X_blocks_test      = X_for_ev,
        W_all              = W_active,
        P_all              = P_active,
        deflate            = TRUE,
        performance_metric = pv$performance_metric,
        correlation_method = pv$correlation_method,
        loading_source     = "train"
      )

      use_frob = identical(pv$performance_metric, "frobenius")
      use_spear = identical(pv$correlation_method, "spearman")

      val_test = pv$val_test
      val_test_n = pv$val_test_n
      val_test_permute_all = pv$val_test_permute_all

      if (val_test != "none" && B < 2L) {
        stop("Prediction-side validation requires at least two blocks.", call. = FALSE)
      }
      if (val_test == "bootstrap" && nrow(dt) < 10L) {
        stop(sprintf(
          "Prediction-side bootstrap validation requires at least 10 test rows; got %d.",
          nrow(dt)
        ), call. = FALSE)
      }
      if (val_test == "bootstrap" && val_test_n < 2L) {
        stop("Prediction-side bootstrap validation requires at least two replicates.",
          call. = FALSE)
      }
      if (val_test == "bootstrap" &&
        (!is.finite(pv$val_test_alpha) || pv$val_test_alpha <= 0 ||
          pv$val_test_alpha >= 1)) {
        stop("Prediction-side bootstrap `val_test_alpha` must be strictly between 0 and 1.",
          call. = FALSE)
      }

      val_test_p = rep(NA_real_, K_active)
      val_test_stat = rep(NA_real_, K_active)
      val_bootstrap_results = NULL # pre-initialize; populated below if val_test="bootstrap"
      val_bootstrap_vectors = NULL
      validation_streams = if (val_test == "none" ||
        is.null(pv$seed_validation)) {
        NULL
      } else {
        mb_rng_streams(K_active, pv$seed_validation)
      }
      run_validation = function(component, fn) {
        if (is.null(validation_streams)) {
          fn()
        } else {
          with_rng_stream_local(validation_streams[[component]], fn)
        }
      }

      score_tables = vector("list", K_active)
      for (k in seq_len(K_active)) {
        Wk = W_active[[k]]
        Tk = matrix(0, nrow(dt), B)
        bi = 0L
        for (bn in block_names) {
          bi = bi + 1L
          w_b = Wk[[bn]]
          cols = colnames(X_cur[[bn]])
          wv = as.numeric(w_b[cols])
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
          res = run_validation(k, function() {
            cpp_perm_test_oos(
              X_test = Xk_list,
              W_trained = Wk,
              n_perm = val_test_n,
              spearman = use_spear,
              frobenius = use_frob,
              permute_all_blocks = isTRUE(val_test_permute_all),
              early_stop_threshold = 1.0
            )
          })
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
          bres = run_validation(k, function() {
            cpp_bootstrap_test_oos(
              X_test = Xk_list,
              W_trained = Wk,
              n_boot = val_test_n,
              spearman = use_spear,
              frobenius = use_frob,
              alpha = pv$val_test_alpha
            )
          })
          observed_correlation = as.numeric(bres$stat_obs %||% NA_real_)
          boot_mean = as.numeric(bres$boot_mean %||% NA_real_)
          boot_bias = as.numeric(bres$bias %||% (boot_mean - observed_correlation))
          boot_se = as.numeric(bres$boot_se %||% NA_real_)
          ci_lower = as.numeric(bres$ci_lower %||% NA_real_)
          ci_upper = as.numeric(bres$ci_upper %||% NA_real_)
          n_boot_done = as.integer(bres$n_boot %||% val_test_n)
          n_boot_requested = as.integer(bres$n_boot_requested %||% val_test_n)
          n_boot_failed = as.integer(bres$n_boot_failed %||%
            (n_boot_requested - n_boot_done))
          conf = as.numeric(bres$confidence_level %||%
            (1 - pv$val_test_alpha))
          interval_type = as.character(bres$interval_type %||% "percentile")
          p_value_note = as.character(bres$p_value_note %||% paste(
            "Not computed: an ordinary bootstrap distribution estimates",
            "uncertainty and is not a null distribution for hypothesis testing."
          ))
          val_test_stat[k] = observed_correlation
          row_k = data.table::data.table(
            component = k,
            estimate = observed_correlation,
            bootstrap_mean = boot_mean,
            bias = boot_bias,
            standard_error = boot_se,
            conf_low = ci_lower,
            conf_high = ci_upper,
            confidence_level = conf,
            interval_type = interval_type,
            replicates_requested = n_boot_requested,
            replicates_effective = n_boot_done,
            replicates_failed = n_boot_failed,
            p_value = NA_real_,
            p_value_note = p_value_note,
            # Backward-compatible descriptive aliases. The former p-value
            # alias remains present but deliberately has no numeric value.
            observed_correlation = observed_correlation,
            boot_mean = boot_mean,
            boot_se = boot_se,
            boot_p_value = NA_real_,
            boot_ci_lower = ci_lower,
            boot_ci_upper = ci_upper,
            n_boot = n_boot_done
          )
          if (is.null(val_bootstrap_vectors)) {
            val_bootstrap_vectors = vector("list", K_active)
          }
          val_bootstrap_vectors[[k]] = as.numeric(bres$replicates %||% numeric())
          val_bootstrap_results = if (is.null(val_bootstrap_results)) {
            row_k
          } else {
            rbind(val_bootstrap_results, row_k, fill = TRUE)
          }
        }

        # Deflate for next component using the stored loadings
        if (k < K_active) {
          Pk = P_active[[k]]
          for (bi in seq_along(block_names)) {
            bn = block_names[[bi]]
            pb = Pk[[bn]]
            if (is.null(pb)) {
              stop(sprintf("Prediction loadings are missing for component %d, block '%s'.", k, bn), call. = FALSE)
            }
            # Use named column access (not positional) for robustness
            t_bn = score_tables[[k]][[paste0("LV", k, "_", bn)]]
            X_cur[[bn]] = X_cur[[bn]] - matrix(t_bn, ncol = 1L) %*% t(as.matrix(pb))
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
          weights_source = used_source,
          time = Sys.time()
        )
        if (pv$val_test != "none") {
          names(val_test_stat) = comp_names
          payload$val_test_stat = val_test_stat
          if (pv$val_test == "permutation") {
            names(val_test_p) = comp_names
            payload$val_test_p = val_test_p
            payload$val_test_params = list(
              n_perm = pv$val_test_n,
              alternative = "greater",
              correction = "(b + 1) / (B + 1)",
              permute_all_blocks = isTRUE(val_test_permute_all),
              seed = pv$seed_validation,
              rng_streams = validation_streams,
              scope = paste(
                "Conditional association diagnostic with fixed trained",
                "weights; not a full-pipeline permutation test."
              )
            )
          } else if (pv$val_test == "bootstrap") {
            payload$val_bootstrap = val_bootstrap_results
            payload$val_boot_vectors = val_bootstrap_vectors
            payload$val_test_params = list(
              n_boot = pv$val_test_n,
              alpha = pv$val_test_alpha,
              interval_type = "percentile",
              seed = pv$seed_validation,
              rng_streams = validation_streams,
              scope = "Descriptive uncertainty; no null-hypothesis p-value."
            )
          }
        }
        payload$run_id = st$run_id %||% self$state$run_id %||% NULL
        log_env_store_last(log_env, payload, run_id = payload$run_id)
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
