#' Multi-Block Sparse Partial Least Squares (MB-sPLS) Transformer
#'
#' @title PipeOp \code{mbspls}: extract up to \code{ncomp} orthogonal latent
#'   components from multiple data blocks
#'
#' @description
#' \strong{PipeOpMBsPLS} fits \emph{sequential} MB-sPLS models and appends one
#' latent variable (LV) per block and component to the task’s backend.
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
#' validation tests} can be requested to quantify out-of-sample support for each
#' extracted component (permutation or bootstrap; see Parameters).
#'
#' The operator is a pure transformer: it performs no internal resampling,
#' tuning or preprocessing. Hyper-parameters such as the L¹ sparsity levels
#' \eqn{c_\mathrm{block}} are tuned externally (e.g., with \pkg{mlr3tuning}).
#'
#' @section State (after training):
#' \describe{
#'   \item{\code{blocks}}{Named list mapping block names to feature column IDs.}
#'   \item{\code{weights}}{List of length \code{ncomp}; each element contains
#'         the block-specific weight vectors \eqn{w_b^{(k)}}.}
#'   \item{\code{loadings}}{List of block loadings \eqn{p_b^{(k)}} used for deflation.}
#'   \item{\code{ncomp}}{Number of components actually retained.}
#'   \item{\code{obj_vec}}{Vector of objective values (MAC/Frobenius) per component (training).}
#'   \item{\code{latent_cor_train}}{Objective value of the last retained component (training).}
#'   \item{\code{ev_block}}{Training explained variance per block (rows = components,
#'         cols = blocks).}
#'   \item{\code{ev_comp}}{Training explained variance per component (summed across blocks).}
#'   \item{\code{p_values}}{Permutation p-values per component if enabled during training.}
#'   \item{\code{performance_metric}}{\code{"mac"} or \code{"frobenius"}.}
#'   \item{\code{c_matrix}}{If provided/derived, the block-by-component sparsity matrix.}
#'   \item{\code{T_mat}}{Training score matrix (per-component deflation applied);
#'         columns ordered \code{LV1_<block1>, ..., LV1_<blockB>, LV2_<block1>, ...}.}
#'   \item{\code{weights_ci}}{(Optional) Bootstrap summary table for weights:
#'         mean/SD and percentile CIs (unconditional and conditional on \eqn{|w|>0}).}
#'   \item{\code{weights_selectfreq}}{(Optional) Selection frequencies \eqn{\in [0,1]}
#'         per feature/block/component.}
#'   \item{\code{weights_boot_draws}}{(Optional) Long-format weight draws if
#'         \code{boot_keep_draws = TRUE}.}
#' }
#'
#' @section Prediction-side logging (\code{log_env$last}):
#' A list containing:
#' \itemize{
#'   \item \code{mac_comp}: numeric vector (length \code{ncomp}) with test MAC/Frobenius per component,
#'   \item \code{ev_block}: matrix \code{(ncomp × n_blocks)} with test per-block explained variances,
#'   \item \code{ev_comp}: numeric vector \code{(ncomp)} with test per-component EV (summed across blocks),
#'   \item \code{T_mat}: test scores \code{(n_test × (ncomp * n_blocks))} with the same column order as training,
#'   \item \code{blocks}: character vector with block names,
#'   \item \code{perf_metric}: objective used (\code{"mac"} or \code{"frobenius"}),
#'   \item \code{time}: POSIXct timestamp,
#'   \item \code{val_test_p}: (if \code{val_test = "permutation"} or \code{"bootstrap"})
#'         per-component p-values on test data (vector, length \code{ncomp}),
#'   \item \code{val_test_stat}: (if available) observed test statistic per component
#'         (latent correlation on test),
#'   \item \code{val_bootstrap}: (if \code{val_test = "bootstrap"}) data.table with
#'         observed statistic, bootstrap mean/SE, p-value, and CI per component,
#'         plus the effective \code{confidence_level} and \code{n_boot}.
#' }
#'
#' @section Parameters:
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
#' @param perm_alpha \code{numeric(1)}. Significance level for training-time
#'   permutation testing.
#' @param c_<block> \code{numeric(1)}. One L¹ sparsity limit per block;
#'   upper bound defaults to \eqn{\sqrt{p_b}}.
#' @param c_matrix \code{matrix}. Optional matrix of L¹ limits with one row per
#'   block and one column per component (overrides \code{ncomp}).
#' @param bootstrap_test \code{logical(1)}. If \code{TRUE}, run a bootstrap on
#'   the \emph{training data} after fitting to obtain weight selection frequencies
#'   and percentile CIs; results are stored in the state (see above).
#' @param n_boot \code{integer(1)}. Number of bootstrap replicates for training bootstrap.
#' @param boot_alpha \code{numeric(1)}. CI level for training bootstrap (default 0.05 for 95% CI).
#' @param boot_keep_draws \code{logical(1)}. Store long-format bootstrap draws in
#'   \code{$state$weights_boot_draws} (memory intensive).
#' @param boot_store_vectors \code{logical(1)}. If \code{TRUE}, store the actual
#'   bootstrap samples (memory intensive).
#' @param boot_min_selectfreq \code{numeric(1)}. Minimum selection frequency for
#'   features to be retained after bootstrap (default: 0).
#' @param val_test \code{character(1)}. Prediction-side validation test on the current
#'   test data: \code{"none"} (default), \code{"permutation"} (row permutations within blocks;
#'   weights fixed), or \code{"bootstrap"} (nonparametric resampling of test rows).
#' @param val_test_n \code{integer(1)}. Number of permutations / bootstrap replicates for
#'   prediction-side validation.
#' @param val_test_alpha \code{numeric(1)}. Early-stop threshold for permutation p-value
#'   computation and CI level for bootstrap validation (e.g., 0.05 → 95% CI).
#' @param val_test_permute_all \code{logical(1)}. If \code{TRUE}, permute all blocks
#'   in the prediction-side permutation test; for \eqn{B=2}, \code{FALSE} only permutes
#'   block 2 against block 1.
#' @param log_env \code{environment} or \code{NULL}. If not \code{NULL}, the operator
#'   writes the above payload to \code{log_env$last} after each \code{$predict()}.
#'
#' @return
#' A \code{PipeOpMBsPLS} that appends columns \code{LVk_<block>} for
#' \eqn{k = 1,\dots,n_\mathrm{comp}} and every block.
#'
#' @examples
#' # see package vignette for a minimal end-to-end example
#'
#' @family PipeOps
#' @keywords internal
#' @importFrom R6 R6Class
#' @import data.table
#' @importFrom checkmate assert_list
#' @importFrom paradox ps p_int p_lgl p_uty p_dbl p_fct
#' @importFrom mlr3pipelines PipeOpTaskPreproc
#' @export
PipeOpMBsPLS = R6::R6Class(
  "PipeOpMBsPLS",
  inherit = mlr3pipelines::PipeOpTaskPreproc,

  public = list(
    #' @field blocks (`list`)\cr named list of blocks
    blocks = NULL,

    #' @description Create a new PipeOpMBsPLS object.
    #' @param id character(1). Identifier (default: "mbspls").
    #' @param blocks named list. Block → character() of feature names (required).
    #' @param param_vals list. Initial ParamSet values (e.g., ncomp, c_* per block, etc.).
    #' @return A new PipeOpMBsPLS object.
    initialize = function(id = "mbspls",
                          blocks,
                          param_vals = list()) {

      checkmate::assert_list(blocks, types = "character",
                             min.len = 1L, names = "unique")

      # ParamSet ---------------------------------------------------------------
      base_params = list(
        blocks              = p_uty(tags = "train", default = blocks),
        ncomp               = p_int(lower = 1L, default = 1L, tags = "train"),
        efficient           = p_lgl(default = FALSE, tags = "train"),
        correlation_method  = p_fct(c("pearson", "spearman"),
                                   default = "pearson", tags = c("train","predict")),
        performance_metric  = p_fct(c("mac", "frobenius"),
                                   default = "mac", tags = c("train","predict")),
        permutation_test    = p_lgl(default = FALSE, tags = "train"),
        n_perm              = p_int(lower = 1L, default = 100L, tags = "train"),
        perm_alpha          = p_dbl(lower = 0, upper = 1, default = 0.05, tags = "train"),
        bootstrap_test      = p_lgl(default = FALSE, tags = "train"),
        boot_alpha        = p_dbl(lower = 0, upper = 1, default = 0.05, tags = "train"),
        boot_keep_draws   = p_lgl(default = TRUE, tags = "train"),
        boot_store_vectors  = p_lgl(default = FALSE, tags = "train"),   # store list-of-vectors per feature
        boot_min_selectfreq = p_dbl(lower = 0, upper = 1, default = 0, tags = "train"),  # post-hoc filter
        c_matrix            = p_uty(tags = c("train", "tune"), default = NULL),
        n_boot              = p_int(lower = 1L, default = 500L, tags = "train"),
        val_test            = p_fct(c("none", "permutation", "bootstrap"), default = "none", tags = "predict"),
        val_test_alpha      = p_dbl(lower = 0, upper = 1, default = 0.05, tags = "predict"),
        val_test_n          = p_int(lower = 1L, default = 1000L, tags = "predict"),
        val_test_permute_all = p_lgl(default = TRUE, tags = "predict"),
        log_env             = p_uty(tags = c("train","predict"), default = NULL)
      )

      # one numeric sparsity hyperparam per block ------------------------------
      lgr$debug("Adding sparsity parameters for blocks: %s",
                paste(names(blocks), collapse = ", "))
      for (bn in names(blocks)) {
        p <- length(blocks[[bn]])
        base_params[[paste0("c_", bn)]] <- p_dbl(
          lower   = 1,
          upper   = sqrt(p),
          default = max(1, sqrt(p)/3),
          tags    = c("train", "tune")
        )
      }

      # optional c_matrix provided via param_vals ------------------------------
      if (!is.null(param_vals$c_matrix)) {
        cm <- param_vals$c_matrix
        checkmate::assert_matrix(cm, mode = "numeric", any.missing = FALSE)
        if (nrow(cm) != length(blocks))
          stop(sprintf("c_matrix must have %d rows (blocks); got %d",
                       length(blocks), nrow(cm)))
        if (ncol(cm) < 1L)
          stop("c_matrix must have at least one column")
        if (!is.null(rownames(cm)))
          cm <- cm[names(blocks), , drop = FALSE]
        param_vals$c_matrix <- cm
        param_vals$ncomp    <- ncol(cm)
      }

      super$initialize(
        id         = id,
        param_set  = do.call(ps, base_params),
        param_vals = param_vals
      )

      self$packages <- "mlr3mbspls"
      self$blocks   <- blocks
    }
  ),

  private = list(

    # --- expand block map to *preprocessed* names once -----------------
    .expand_block_cols = function(dt_names, cols) {
      esc <- function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\1", s)
      unique(unlist(lapply(cols, function(co) {
        if (co %in% dt_names) co else grep(paste0("^", esc(co), "(\\.|$)"), dt_names, value = TRUE)
      })))
    },

    # ------------------------------- train -----------------------------------
    .train_dt = function(dt, levels, target = NULL) {
      pv <- utils::modifyList(
        paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "train"),
        keep.null = TRUE
      )
      use_frob <- (pv$performance_metric == "frobenius")
      blocks <- pv$blocks

      # validate + expand blocks against *current* dt (after upstream pipeops)
      dt_names <- names(dt)
      blocks <- lapply(pv$blocks, function(cols) {
        cand <- private$.expand_block_cols(dt_names, cols)
        # keep only numeric, non-constant columns
        cand <- cand[vapply(cand, function(cl) is.numeric(dt[[cl]]), logical(1))]
        if (!length(cand)) return(character(0))
        keep <- vapply(cand, function(cl) stats::var(dt[[cl]], na.rm = TRUE) > 0, logical(1))
        cand[keep]
      })
      blocks <- Filter(length, blocks)
      if (length(blocks) == 0)
        stop("No block contains at least one numeric, non-constant feature.")

      n_block <- length(blocks)
      X_list <- lapply(blocks, \(cols) {
        mat <- as.matrix(dt[, ..cols])
        storage.mode(mat) <- "double"
        mat
      })

      # handle c_matrix vs c_vec
      if (!is.null(pv$c_matrix)) {
        cm <- pv$c_matrix
        checkmate::assert_matrix(cm, mode = "numeric", any.missing = FALSE)
        if (nrow(cm) != n_block)
          stop(sprintf("c_matrix must have %d rows (blocks); got %d", n_block, nrow(cm)))
        if (!is.null(rownames(cm))) {
          missing_rows <- setdiff(names(blocks), rownames(cm))
          if (length(missing_rows))
            stop("Row names of c_matrix do not cover all blocks: ",
                 paste(missing_rows, collapse = ", "))
          cm <- cm[names(blocks), , drop = FALSE]
        }
        pv$ncomp <- ncol(cm)
        c_matrix <- cm
        c_vec    <- NULL
        lgr$info("Fitting MB-sPLS with %d blocks, %d components, and c-constraints (matrix).",
                 n_block, pv$ncomp)
      } else {
        c_vec <- vapply(names(blocks), \(bn) pv[[ paste0("c_", bn) ]], numeric(1))
        c_matrix <- NULL
        lgr$info("Fitting MB-sPLS with %d blocks, %d components, and c-constraints: %s",
                 n_block, pv$ncomp, paste(c_vec, collapse = ", "))
      }

      # fit in C++
      fit <- if (is.null(c_matrix)) {
        cpp_mbspls_multi_lv(
          X_blocks      = X_list,
          c_constraints = c_vec,
          K             = pv$ncomp,
          max_iter      = 1000L,
          spearman      = (pv$correlation_method == "spearman"),
          do_perm       = isTRUE(pv$permutation_test),
          n_perm        = pv$n_perm,
          alpha         = pv$perm_alpha,
          frobenius     = use_frob
        )
      } else {
        mbspls_multi_lv_matrix(
          X_blocks  = X_list,
          c_matrix  = c_matrix,
          max_iter  = 1000L,
          max_tol   = 1e-6,
          spearman  = (pv$correlation_method == "spearman"),
          do_perm   = isTRUE(pv$permutation_test),
          n_perm    = pv$n_perm,
          alpha     = pv$perm_alpha,
          frobenius = use_frob
        )
      }

      self$state$c_matrix <- c_matrix

      lgr$info("C++ returned %d component(s)", length(fit$W))
      lgr$info("Objectives per component: %s",
               paste(round(fit$objective, 4), collapse = ", "))
      if (!is.null(fit$p_values))
        lgr$info("Permutation p-values: %s",
                 paste(signif(fit$p_values, 3), collapse = ", "))
      if (length(fit$W) == 0)
        stop("No components extracted – check sparsity settings.")

      W_all   <- fit$W
      P_all   <- fit$P
      obj     <- fit$objective
      pvals   <- fit$p_values
      ev_blk  <- fit$ev_block
      ev_cmp  <- fit$ev_comp

      n_kept  <- length(W_all)
      B       <- length(blocks)
      block_names <- names(blocks)
      comp_names  <- sprintf("LC_%02d", seq_len(n_kept))

      # --- pad / name W and P (leave as-is) ---
      pad_and_name <- function(x, feat_names) {
        if (length(x) == 0L) {
          x <- numeric(length(feat_names))
        } else if (length(x) != length(feat_names)) {
          stop(sprintf("Internal size mismatch: expected %d, got %d",
                       length(feat_names), length(x)))
        }
        stats::setNames(x, feat_names)
      }
      for (k in seq_len(n_kept)) {
        for (b in seq_len(B)) {
          feat_names <- blocks[[block_names[b]]]
          W_all[[k]][[b]] <- pad_and_name(W_all[[k]][[b]], feat_names)
          P_all[[k]][[b]] <- pad_and_name(P_all[[k]][[b]], feat_names)
        }
        names(W_all[[k]]) <- names(P_all[[k]]) <- block_names
      }
      names(W_all) <- names(P_all) <- comp_names

      # ---- NOW run training bootstrap (if enabled) ----
      if (isTRUE(pv$bootstrap_test) && (pv$n_boot %||% 0L) > 0L) {
        lgr$info("Running training bootstrap for weights (B = %d)", pv$n_boot)
        bt <- private$.bootstrap_weights_ci(
          X_list      = X_list,
          blocks      = blocks,
          W_ref       = W_all,          # already named now
          ncomp       = n_kept,
          c_matrix    = c_matrix,
          c_vec       = if (is.null(c_matrix)) c_vec else NULL,
          corr_method = pv$correlation_method,
          perf_metric = pv$performance_metric,
          n_boot      = pv$n_boot,
          alpha       = pv$boot_alpha %||% 0.05,
          keep_draws  = isTRUE(pv$boot_keep_draws) || isTRUE(pv$boot_store_vectors),
          store_vectors = isTRUE(pv$boot_store_vectors)
        )
        self$state$weights_ci         <- bt$summary
        self$state$weights_selectfreq <- bt$select_freq
        if (!is.null(bt$draws))        self$state$weights_boot_draws <- bt$draws # optional long format
        if (!is.null(bt$vectors_map))  self$state$weights_boot_vectors <- bt$vectors_map

        # optional post-hoc stability filter on training weights (store *additionally*)
        thr <- as.numeric(pv$boot_min_selectfreq %||% 0)
        if (is.finite(thr) && thr > 0) {
          W_filt <- lapply(seq_len(n_kept), function(k) {
            wf <- W_all[[k]]
            for (b in names(blocks)) {
              # rows of select_freq are component × block × feature with 'freq' in [0,1]
              sf <- bt$select_freq[component == sprintf("LC_%02d", k) & block == b]
              if (nrow(sf)) {
                drop_feats <- sf$feature[ sf$freq < thr ]
                if (length(drop_feats)) wf[[b]][drop_feats] <- 0
              }
            }
            wf
          })

          names(W_filt) <- names(W_all)
          self$state$weights_stability_filtered <- W_filt
          attr(self$state$weights_stability_filtered, "min_freq") <- thr
          lgr$info("Stored stability-filtered weights at min_freq = %.3f", thr)
        }
      }

      # pad / name W and P
      pad_and_name <- function(x, feat_names) {
        if (length(x) == 0L) {
          x <- numeric(length(feat_names))
        } else if (length(x) != length(feat_names)) {
          stop(sprintf("Internal size mismatch: expected %d, got %d",
                       length(feat_names), length(x)))
        }
        stats::setNames(x, feat_names)
      }
      for (k in seq_len(n_kept)) {
        for (b in seq_len(B)) {
          feat_names <- blocks[[block_names[b]]]
          W_all[[k]][[b]] <- pad_and_name(W_all[[k]][[b]], feat_names)
          P_all[[k]][[b]] <- pad_and_name(P_all[[k]][[b]], feat_names)
        }
        names(W_all[[k]]) <- names(P_all[[k]]) <- block_names
      }
      names(W_all) <- names(P_all) <- comp_names

      # name EV structures
      if (!is.null(ev_blk)) {
        colnames(ev_blk) <- block_names
        rownames(ev_blk) <- comp_names
      }
      if (!is.null(ev_cmp)) names(ev_cmp) <- comp_names

      # ---- RECOMPUTE training scores with proper deflation (robust) ----------
      X_cur <- X_list
      score_tables <- vector("list", n_kept)
      for (k in seq_len(n_kept)) {
        Wk <- W_all[[k]]
        Tk <- matrix(0, nrow(dt), B)
        for (b in seq_len(B)) {
          Tk[, b] <- X_cur[[b]] %*% Wk[[b]]
        }
        score_tables[[k]] <- data.table::as.data.table(Tk)
        data.table::setnames(score_tables[[k]],
          paste0("LV", k, "_", block_names))

        # deflate for next component
        if (k < n_kept) {
          Pk <- P_all[[k]]
          for (b in seq_len(B)) {
            X_cur[[b]] <- X_cur[[b]] - Tk[, b] %*% t(Pk[[b]])
          }
        }
      }
      dt_lat <- do.call(cbind, score_tables)
      T_mat_train <- as.matrix(dt_lat)  # consistent shape = nrow × (B * n_kept)

      # save state
      self$state$blocks             <- blocks
      self$state$weights            <- W_all
      self$state$loadings           <- P_all
      self$state$ncomp              <- n_kept
      self$state$T_mat              <- T_mat_train
      self$state$obj_vec            <- obj
      self$state$p_values           <- pvals
      self$state$ev_block           <- ev_blk
      self$state$ev_comp            <- ev_cmp
      self$state$latent_cor_train   <- tail(obj, 1)
      self$state$performance_metric <- pv$performance_metric

      lgr$info("Training done; last latent correlation = %.4f", tail(obj, 1))
      dt_lat
    },

    # ------------------------------- predict ---------------------------------
    .predict_dt = function(dt, levels, target = NULL) {
      pv <- utils::modifyList(
        paradox::default_values(self$param_set),
        self$param_set$get_values(tags = "predict"),
        keep.null = TRUE
      )

      st   <- self$state
      B    <- length(st$blocks)
      n_k  <- st$ncomp

      # ensure all trained columns exist
      missing_cols <- setdiff(unlist(st$blocks), names(dt))
      if (length(missing_cols)) {
        lgr$warn("Adding %d feature columns (all-zero) that were present during training",
                 length(missing_cols))
        dt[, (missing_cols) := 0.0]
      }

      # build current X (test)
      X_cur <- lapply(st$blocks, \(cols) {
        mat <- as.matrix(dt[, ..cols])
        storage.mode(mat) <- "double"
        mat
      })

      # compute test explained variances + objective per component
      test_ev_results <- compute_pipeop_test_ev(X_cur, st)
      # expected to return list: $ev_block (K×B), $ev_comp (K), $mac_comp (K)

      use_frob    <- identical(self$state$performance_metric, "frobenius")
      use_spear   <- identical(pv$correlation_method, "spearman")
      val_test    <- pv$val_test
      val_test_n  <- pv$val_test_n
      val_test_permute_all <- pv$val_test_permute_all
      n_k         <- st$ncomp
      B           <- length(st$blocks)

      # holders for validation test results
      val_test_p   <- rep(NA_real_, n_k)
      val_test_stat <- rep(NA_real_, n_k)

      # compute test scores with deflation (mirrors training) + optional validation perm test
      score_tables <- vector("list", n_k)
      for (k in seq_len(n_k)) {

        # snapshot the *current* (already deflated up to k-1) test blocks for validation testing
        Xk_list <- lapply(X_cur, function(x) { storage.mode(x) <- "double"; x })

        # scores for comp k
        Wk <- st$weights[[k]]
        Tk <- matrix(0, nrow(dt), B)
        for (b in seq_len(B)) {
          Tk[, b] <- X_cur[[b]] %*% Wk[[b]]
        }
        score_tables[[k]] <- data.table::as.data.table(Tk)
        data.table::setnames(score_tables[[k]], paste0("LV", k, "_", names(st$blocks)))

        # ----- Validation tests for component k (optional) -----
        if (val_test == "permutation" && B >= 2L) {
          # coerce to lists expected by C++
          Xk_cpp <- Xk_list
          Wk_cpp <- Wk

          res <- cpp_perm_test_oos(
            X_test = Xk_cpp,
            W_trained = Wk_cpp,
            n_perm = pv$val_test_n,
            spearman = use_spear,
            frobenius = use_frob,
            permute_all_blocks = isTRUE(pv$val_test_permute_all),
            early_stop_threshold = pv$val_test_alpha
          )

          if (is.list(res)) {
            val_test_p[k]    <- as.numeric(res$p_value %||% NA_real_)
            val_test_stat[k] <- as.numeric(res$stat_obs %||% NA_real_)
          } else {
            val_test_p[k]    <- as.numeric(res)
            val_test_stat[k] <- NA_real_
          }
          lgr$info("Validation permutation test for component %d: p-value = %.3f, stat = %.3f",
                   k, val_test_p[k], val_test_stat[k])
        }

        if (val_test == "bootstrap") {
          # Bootstrap validation: resample test data and recompute latent correlation
          # This tests stability of the correlation structure on the test set
          
          n_test <- nrow(dt)
          if (n_test < 10) {
            lgr$warn("Insufficient test samples (%d) for bootstrap validation", n_test)
            next
          }
          
          # First, compute the observed correlation on the original test data
          observed_correlation <- 0
          if (B >= 2 && nrow(Tk) >= 3) {
            score_cors <- c()
            for (i in seq_len(B - 1L)) {
              for (j in (i + 1L):B) {
                if (stats::sd(Tk[, i]) > 0 && stats::sd(Tk[, j]) > 0) {
                  r <- stats::cor(Tk[, i], Tk[, j], method = pv$correlation_method)
                  if (is.finite(r)) {
                    score_cors <- c(score_cors, if (use_frob) r^2 else abs(r))
                  }
                }
              }
            }
            
            observed_correlation <- if (length(score_cors) > 0) {
              if (use_frob) sqrt(sum(score_cors)) else mean(score_cors)
            } else 0
          }
          
          boot_correlations <- numeric(val_test_n)
          
          # Run bootstrap replicates
          for (boot_rep in seq_len(val_test_n)) {
            # Bootstrap resample indices
            boot_idx <- sample(seq_len(n_test), replace = TRUE)
            
            # Compute scores for bootstrap sample using trained weights
            Tk_boot <- matrix(0, length(boot_idx), B)
            for (b in seq_len(B)) {
              Tk_boot[, b] <- Xk_list[[b]][boot_idx, , drop = FALSE] %*% Wk[[b]]
            }
            
            # Compute correlation between block scores for this bootstrap sample
            if (B >= 2 && nrow(Tk_boot) >= 3) {
              score_cors <- c()
              for (i in seq_len(B - 1L)) {
                for (j in (i + 1L):B) {
                  if (stats::sd(Tk_boot[, i]) > 0 && stats::sd(Tk_boot[, j]) > 0) {
                    r <- stats::cor(Tk_boot[, i], Tk_boot[, j], method = pv$correlation_method)
                    if (is.finite(r)) {
                      score_cors <- c(score_cors, if (use_frob) r^2 else abs(r))
                    }
                  }
                }
              }
              
              boot_correlations[boot_rep] <- if (length(score_cors) > 0) {
                if (use_frob) sqrt(sum(score_cors)) else mean(score_cors)
              } else 0
            } else {
              boot_correlations[boot_rep] <- 0
            }
          }
          
          # Compute bootstrap statistics
          boot_mean <- mean(boot_correlations, na.rm = TRUE)
          boot_se <- stats::sd(boot_correlations, na.rm = TRUE)
          
          # Calculate p-value: proportion of bootstrap values <= observed value
          # This tests H0: true correlation <= observed (one-sided test for significance)
          valid_boot <- boot_correlations[is.finite(boot_correlations)]
          boot_p_value <- if (length(valid_boot) > 0) {
            mean(valid_boot <= observed_correlation)
          } else NA_real_

          # Calculate confidence interval based on val_test_alpha parameter
          # val_test_alpha = 0.05 corresponds to 95% confidence level
          confidence_level <- 1 - (pv$val_test_alpha %||% 0.05)
          z_value <- stats::qnorm(1 - (1 - confidence_level) / 2)
          ci_lower <- boot_mean - z_value * boot_se
          ci_upper <- boot_mean + z_value * boot_se
          
          lgr$info("Bootstrap validation for component %d: obs=%.4f, mean=%.4f, SE=%.4f, p=%.4f, %g%% CI=[%.4f, %.4f]",
                   k, observed_correlation, boot_mean, boot_se, boot_p_value, confidence_level * 100, ci_lower, ci_upper)
          
          # Store bootstrap validation results for logging
          boot_results <- data.table::data.table(
            component = k,
            observed_correlation = observed_correlation,
            boot_mean = boot_mean,
            boot_se = boot_se,
            boot_p_value = boot_p_value,
            boot_ci_lower = ci_lower,
            boot_ci_upper = ci_upper,
            confidence_level = confidence_level,
            n_boot = val_test_n
          )
          
          # Store p-value for compatibility with permutation test reporting
          val_test_p[k] <- boot_p_value
          val_test_stat[k] <- observed_correlation

          # Accumulate results across components
          if (k == 1L) {
            val_bootstrap_results <- boot_results
          } else {
            val_bootstrap_results <- rbind(val_bootstrap_results, boot_results, fill = TRUE)
          }
        }

        # ------------------------------------------------------------

        # deflate for next component
        if (k < n_k) {
          Pk <- st$loadings[[k]]
          for (b in seq_len(B)) {
            X_cur[[b]] <- X_cur[[b]] - Tk[, b] %*% t(Pk[[b]])
          }
        }
      }

      dt_lat     <- do.call(cbind, score_tables)
      T_mat_test <- as.matrix(dt_lat)

      # existing coerce/logging...
      ev_block_test <- as.matrix(test_ev_results$ev_block)
      ev_comp_test  <- as.numeric(test_ev_results$ev_comp)
      mac_comp_test <- as.numeric(test_ev_results$mac_comp)
      comp_names    <- sprintf("LC_%02d", seq_len(n_k))
      colnames(ev_block_test) <- names(st$blocks)
      rownames(ev_block_test) <- comp_names
      names(ev_comp_test)     <- comp_names
      names(mac_comp_test)    <- comp_names

      latent_cor_test <- if (length(mac_comp_test)) tail(mac_comp_test, 1) else NA_real_

      # add validation results to logging payload if requested
      log_env <- self$param_set$values$log_env
      if (!is.null(log_env) && inherits(log_env, "environment")) {
        payload <- list(
          mac_comp    = mac_comp_test,
          ev_block    = ev_block_test,
          ev_comp     = ev_comp_test,
          T_mat       = T_mat_test,
          blocks      = names(self$state$blocks),
          perf_metric = self$state$performance_metric,
          time        = Sys.time()
        )
        
        # Add permutation test results if available
        if (val_test == "permutation" && exists("val_test_p", inherits = FALSE)) {
          names(val_test_p) <- comp_names
          names(val_test_stat) <- comp_names
          payload$val_test_p <- val_test_p
          payload$val_test_stat <- val_test_stat
          payload$val_test_params <- list(
            n_perm = pv$val_test_n,
            alpha = pv$val_test_alpha,
            permute_all_blocks = isTRUE(pv$val_test_permute_all)
          )
          lgr$info("Prediction validation permutation p-values: %s",
                   paste(signif(val_test_p, 4), collapse = ", "))
        }
        
        # Add bootstrap test results if available
        if (val_test == "bootstrap" && exists("val_bootstrap_results", inherits = FALSE)) {
          payload$val_bootstrap <- val_bootstrap_results
          lgr$info("Prediction validation bootstrap means: %s",
                   paste(round(val_bootstrap_results$boot_mean, 4), collapse = ", "))
          lgr$info("Prediction validation bootstrap p-values: %s",
                   paste(signif(val_bootstrap_results$boot_p_value, 4), collapse = ", "))
        }
        
        log_env$last <- payload
      }

      lgr$info("Prediction latent correlation (last comp.) = %.4f", latent_cor_test)
      lgr$info("Eval explained variance (component-wise): %s",
               paste(round(ev_comp_test, 4), collapse = ", "))
      lgr$info("Eval %s (component-wise): %s",
               if (st$performance_metric == "frobenius") "Frobenius" else "MAC",
               paste(round(mac_comp_test, 4), collapse = ", "))

      dt_lat
    },

    # for hashing
    .additional_phash_input = function() {
      list(
        blocks    = self$param_set$values$blocks,
        efficient = self$param_set$values$efficient
      )
    },

    #' Compute bootstrap-based latent correlation between data blocks
    #' 
    #' This function calculates a summary correlation metric between different data blocks
    #' based on their weight vectors from bootstrap replicates. It measures how strongly
    #' the latent components from different blocks are correlated.
    .compute_bootstrap_latent_cor = function(boot_slice, use_frob = FALSE) {
      
      # Early exit: need at least 3 observations for meaningful correlation
      if (nrow(boot_slice) < 3L) return(0)
      
      # Get correlation method from parameter set (default: "pearson")
      corr_method <- self$param_set$values$correlation_method %||% "pearson"
      
      # Split weights by block to get per-block weight vectors
      weights_by_block <- split(boot_slice$weights, boot_slice$block)
      
      # Filter out blocks with insufficient or constant data
      # Each block needs: ≥2 finite values AND non-zero variance
      is_block_valid <- vapply(weights_by_block, function(block_weights) {
        n_finite <- sum(is.finite(block_weights))
        has_variance <- n_finite >= 2 && stats::var(block_weights, na.rm = TRUE) > 0
        return(has_variance)
      }, logical(1))
      
      weights_by_block <- weights_by_block[is_block_valid]
      n_blocks <- length(weights_by_block)
      
      # Need at least 2 blocks for pairwise correlations
      if (n_blocks < 2) return(0)
      
      # Compute pairwise correlations between all block pairs
      correlation_values <- c()
      block_names <- names(weights_by_block)
      
      # Nested loop over all unique block pairs (i, j) where i < j
      for (i in seq_len(n_blocks - 1L)) {
        for (j in (i + 1L):n_blocks) {
          
          # Match vector lengths (take minimum length to avoid index errors)
          block_i_weights <- weights_by_block[[i]]
          block_j_weights <- weights_by_block[[j]]
          min_length <- min(length(block_i_weights), length(block_j_weights))
          
          # Skip if insufficient data for correlation
          if (min_length < 2) next
          
          # Extract matched-length vectors
          vec_i <- block_i_weights[seq_len(min_length)]
          vec_j <- block_j_weights[seq_len(min_length)]
          
          # Skip if either vector has zero variance (would cause cor() to fail)
          if (stats::sd(vec_i) == 0 || stats::sd(vec_j) == 0) next
          
          # Compute correlation between the two block weight vectors
          correlation <- stats::cor(vec_i, vec_j, method = corr_method)
          
          # Store valid correlations (transform based on metric choice)
          if (is.finite(correlation)) {
            if (use_frob) {
              # Frobenius: use squared correlation (r²)
              correlation_values <- c(correlation_values, correlation^2)
            } else {
              # MAC: use absolute correlation (|r|)
              correlation_values <- c(correlation_values, abs(correlation))
            }
          }
        }
      }
      
      # Aggregate correlations into final metric
      if (!length(correlation_values)) return(0)
      
      if (use_frob) {
        # Frobenius norm: sqrt of sum of squared correlations
        return(sqrt(sum(correlation_values)))
      } else {
        # Mean Absolute Correlation: average of |r| values
        return(mean(correlation_values))
      }
    },

    # Bootstrap weights on training data; returns summaries for CIs
    .bootstrap_weights_ci = function(
      X_list, blocks, W_ref, ncomp,
      c_matrix = NULL, c_vec = NULL,
      corr_method = "pearson",
      perf_metric = "mac",
      n_boot = 200L, alpha = 0.05,
      keep_draws = TRUE, store_vectors = FALSE
    ) {
      B  <- length(blocks)
      bn <- names(blocks)
      N  <- nrow(X_list[[1]])
      stopifnot(all(vapply(X_list, nrow, 1L) == N))

      # ---- helper: one fit with same settings as main training (no perms)
      fit_once <- function(Xlst) {
        if (is.null(c_matrix)) {
          cpp_mbspls_multi_lv(
            X_blocks      = Xlst,
            c_constraints = c_vec,
            K             = ncomp,
            max_iter      = 1000L,
            spearman      = (corr_method == "spearman"),
            do_perm       = FALSE,
            n_perm        = 0L,
            alpha         = 0.05,
            frobenius     = (perf_metric == "frobenius")
          )
        } else {
          mbspls_multi_lv_matrix(
            X_blocks  = Xlst,
            c_matrix  = c_matrix,
            max_iter  = 1000L,
            max_tol   = 1e-6,
            spearman  = (corr_method == "spearman"),
            do_perm   = FALSE,
            n_perm    = 0L,
            alpha     = 0.05,
            frobenius = (perf_metric == "frobenius")
          )
        }
      }

      # ---- reference scores for orientation
      T_ref <- lapply(seq_len(ncomp), function(k)
        lapply(seq_len(B), function(b) as.numeric(X_list[[b]] %*% W_ref[[k]][[bn[b]]]))
      )

      # ---- convenience helpers
      pad_to_order <- function(w_boot, w_ref_named) {
        all_feat <- names(w_ref_named)
        out <- numeric(length(all_feat)); names(out) <- all_feat
        if (!is.null(names(w_boot))) out[names(w_boot)] <- w_boot
        out
      }
      comp_lab <- sprintf("LC_%02d", seq_len(ncomp))
      n_eff <- integer(ncomp)  # effective bootstrap replicates per component

      # ---- selection-frequency table (character keys → safe joins)
      sel_freq <- data.table::rbindlist(
        lapply(seq_len(ncomp), function(k)
          data.table::rbindlist(
            lapply(bn, function(b)
              data.table::data.table(
                component = comp_lab[k], block = b,
                feature   = names(W_ref[[k]][[b]]), sel = 0L
              )
            ), use.names = TRUE, fill = TRUE
          )
        ), use.names = TRUE, fill = TRUE
      )
      data.table::setkey(sel_freq, component, block, feature)

      # ---- storage
      draws_list <- if (isTRUE(keep_draws)) vector("list", n_boot) else NULL
      if (isTRUE(store_vectors)) vectors_map <- lapply(seq_len(ncomp), function(i)
        stats::setNames(rep(list(list()), length(bn)), bn)) else vectors_map <- NULL

      # helper to append values to vectors_map efficiently
      .vec_append <- function(k, b, features, values) {
        vm <- vectors_map[[k]][[b]]
        for (i in seq_along(features)) {
          f <- features[i]
          v <- values[i]
          if (is.null(vm[[f]])) vm[[f]] <- v else vm[[f]] <- c(vm[[f]], v)
        }
        vectors_map[[k]][[b]] <<- vm
      }
      accum <- list()
      keyfun <- function(k, b, f) paste(k, b, f, sep = "||")

      # ---- bootstrap loop
      orient_min_cor <- 0.05
      for (r in seq_len(n_boot)) {
        if (r %% 50 == 0) lgr$info(" Bootstrap replicate %d / %d", r, n_boot)
        idx <- sample.int(N, replace = TRUE)
        Xb  <- lapply(X_list, function(X) X[idx, , drop = FALSE])
        fit_r <- fit_once(Xb); W_r <- fit_r$W
        if (length(W_r) < ncomp) next

        for (k in seq_len(ncomp)) {
          # average score correlation across blocks to decide orientation
          T_boot <- lapply(seq_len(B), function(b)
            as.numeric(Xb[[b]] %*% W_r[[k]][[b]])
          )
          cor_vec <- vapply(seq_len(B), function(b) {
            suppressWarnings(stats::cor(T_boot[[b]], T_ref[[k]][[b]][idx]))
          }, numeric(1))
          cor_vec[!is.finite(cor_vec)] <- 0
          mean_cor <- mean(cor_vec)
          if (!is.finite(mean_cor) || abs(mean_cor) < orient_min_cor) next
          n_eff[k] <- n_eff[k] + 1L
          s_k <- if (mean_cor >= 0) 1 else -1

          for (b in seq_len(B)) {
            block_name <- bn[b]
            w_b <- W_r[[k]][[b]]
            if (is.null(names(w_b)) &&
                length(w_b) == length(names(W_ref[[k]][[block_name]])))
              names(w_b) <- names(W_ref[[k]][[block_name]])

            w_aligned <- s_k * pad_to_order(w_b, W_ref[[k]][[block_name]])

            # selection frequency (vectorised join)
            inc_dt <- data.table::data.table(
              component = comp_lab[k], block = block_name,
              feature = names(w_aligned),
              inc = as.integer(abs(w_aligned) > 0)
            )
            sel_freq[inc_dt, sel := sel + i.inc, on = .(component, block, feature)]
            if (isTRUE(store_vectors)) .vec_append(k, block_name, names(w_aligned), as.numeric(w_aligned))

            # streaming moments (when not storing draws)
            for (i in seq_along(w_aligned)) {
              ky <- keyfun(k, block_name, names(w_aligned)[i])
              val <- w_aligned[i]
              if (is.null(accum[[ky]])) {
                accum[[ky]] <- list(n = 1L, mean = val, m2 = 0)
              } else {
                a <- accum[[ky]]
                a$n <- a$n + 1L
                delta <- val - a$mean
                a$mean <- a$mean + delta / a$n
                a$m2   <- a$m2 + delta * (val - a$mean)
                accum[[ky]] <- a
              }
            }

            if (isTRUE(keep_draws)) {
              draws_list[[r]] <- data.table::rbindlist(list(
                draws_list[[r]],
                data.table::data.table(
                  replicate = r, component = k, block = block_name,
                  feature = names(w_aligned), weight = as.numeric(w_aligned)
                )
              ), use.names = TRUE, fill = TRUE)
            }
          }
        }
      }

      # finalise selection freq in [0,1]
      if (nrow(sel_freq)) {
        sel_freq[, eff := n_eff[as.integer(gsub("^LC_", "", component))]]
        sel_freq[eff <= 0, eff := NA_integer_]
        sel_freq[, `:=`(freq = sel / eff, sel = NULL, eff = NULL)]
      }
      sel_freq$component <- as.character(sel_freq$component)

      # ---- summarise to CIs
      if (isTRUE(keep_draws)) {
        draws <- data.table::rbindlist(draws_list, use.names = TRUE, fill = TRUE)
        draws[, component := factor(sprintf("LC_%02d", component),
                                    levels = sprintf("LC_%02d", seq_len(ncomp)))]
        draws[, block := factor(block, levels = names(blocks))]

        a <- alpha %||% 0.05
        summary <- draws[, {
          nz <- abs(weight) > 0
          list(
            boot_mean   = mean(weight, na.rm = TRUE),
            boot_sd     = stats::sd(weight, na.rm = TRUE),
            ci_lower    = stats::quantile(weight, probs = a/2,     na.rm = TRUE),
            ci_upper    = stats::quantile(weight, probs = 1 - a/2, na.rm = TRUE),
            ci_lower_nz = if (any(nz)) stats::quantile(weight[nz], probs = a/2,     na.rm = TRUE) else NA_real_,
            ci_upper_nz = if (any(nz)) stats::quantile(weight[nz], probs = 1 - a/2, na.rm = TRUE) else NA_real_
          )
        }, by = .(component, block, feature)]
      } else {
        # normal-approx CIs from streaming moments; conditional CIs unavailable
        rows <- vector("list", length(accum)); ii <- 0L
        z <- stats::qnorm(1 - alpha/2)
        for (ky in names(accum)) {
          ii <- ii + 1L; a <- accum[[ky]]
          parts <- strsplit(ky, "\\|\\|")[[1]]
          k <- as.integer(parts[1]); b <- parts[2]; f <- parts[3]
          sd <- if (a$n > 1L) sqrt(a$m2 / (a$n - 1L)) else 0
          lwr <- a$mean - z * sd; upr <- a$mean + z * sd
          rows[[ii]] <- data.frame(
            component = sprintf("LC_%02d", k),
            block = b, feature = f,
            boot_mean = a$mean, boot_sd = sd,
            ci_lower = lwr, ci_upper = upr,
            ci_lower_nz = NA_real_, ci_upper_nz = NA_real_, # <- present, but NA
            stringsAsFactors = FALSE
          )
        }
        summary <- do.call(rbind, rows)
      }

      # merge selection frequencies
      summary <- merge(as.data.frame(summary), as.data.frame(sel_freq),
                       by = c("component","block","feature"), all.x = TRUE)

      # order factors last (for plotting facets)
      summary$component <- factor(summary$component, levels = comp_lab)
      summary$block     <- factor(summary$block,     levels = bn)

      list(summary = summary,
        select_freq = sel_freq,
        draws = if (exists("draws")) draws else NULL,
        vectors_map = if (!is.null(vectors_map) && isTRUE(store_vectors)) {
          # name outer lists for readability
          names(vectors_map) <- comp_lab
          for (k in seq_len(ncomp)) vectors_map[[k]] <- stats::setNames(vectors_map[[k]], bn)
          vectors_map
        } else NULL)
    }
  )
)