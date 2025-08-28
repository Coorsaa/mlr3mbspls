#' Sequential Component-wise Tuner for MB-sPLS
#'
#' @title TunerSeqMBsPLS
#'
#' @description
#' **`TunerSeqMBsPLS`** performs *sequential* hyper-parameter optimisation of
#' a multi-block sparse PLS (MB-sPLS) model in the **mlr3** ecosystem.
#'
#' For each latent component (LC) it:
#' \enumerate{
#'   \item samples candidate block-sparsity vectors \eqn{c = (c_1,\dots,c_B)},
#'   \item scores them by inner resampling using a correlation objective
#'         (mean absolute correlation or Frobenius norm of block scores),
#'   \item picks the best candidate, refits that LC on the current full-data
#'         residuals, (optionally) applies a permutation test, and deflates,
#'   \item repeats until \code{ncomp} components are extracted or the test
#'         is not significant (LC1 is always kept).
#' }
#'
#' **Measure handling.** The tuner **does not** assume a specific measure ID.
#' Whatever measure you pass to the tuning instance (e.g. `msr("mbspls.mac")`,
#' `msr("mbspls.mac.lv1")`, …) is used to **name the result automatically**.
#' If that measure has `minimize = TRUE`, the returned value is negated so the
#' instance receives the correct orientation.
#'
#' @section Leakage-safe preprocessing:
#' The tuner extracts and resamples components using the preprocessing steps
#' found **upstream** of \code{po("mbspls")} in the supplied
#' \code{mlr3::GraphLearner}. These steps are trained only on the respective
#' fold’s training data and then applied to its validation data to avoid any
#' leakage into validation folds.
#'
#' @section Block mapping & encoding:
#' MB-sPLS operates on **numeric matrices**. If your preprocessing includes
#' factor encoding (e.g., `PipeOpEncode(method = "treatment")`), original
#' feature names may be expanded/renamed (e.g., `X` → `X.a`, `X.b`, ...).
#' Therefore, the \strong{block map passed to \code{po("mbspls")}} must refer
#' to the \emph{post-preprocessing} column names. In user code, build your
#' graph with a helper like \code{expand_blocks_after_pre()} and
#' \code{make_mbs_unsup_graph()} (see the analysis script for Option B),
#' which ensure that both \code{po("sitecorr")} and \code{po("mbspls")} receive
#' an up-to-date block map that matches the preprocessed task.
#'
#' @section Returned result:
#' The optimal per-block sparsity matrix \eqn{C^\* \in \mathbb{R}^{B\times K}}
#' (blocks × components) is returned in
#' \code{tuning_instance$result$learner_param_vals$c_matrix}.
#'
#' @section Usage:
#' \preformatted{
#' tuner  <- TunerSeqMBsPLS$new(
#'   tuner          = "random_search",
#'   budget         = 1500L,
#'   resampling     = rsmp("cv", folds = 3),
#'   parallel       = "none",
#'   early_stopping = TRUE,
#'   n_perm         = 1000L,
#'   perm_alpha     = 0.05,
#'   performance_metric = "mac"
#' )
#'
#' gl <- make_mbs_unsup_graph(task, ncomp = 5L, perf_metric = "mac",
#'                            learner = lrn("clust.kmeans", centers = 1))
#' inst <- mlr3tuning::ti(
#'   task = task,
#'   learner = gl,
#'   resampling = rsmp("holdout"),
#'   measure = msr("mbspls.mac_evwt"),
#'   terminator = trm("evals", n_evals = 1)
#' )
#' tuner$optimize(inst)
#' best_C <- inst$result$learner_param_vals$c_matrix
#' }
#'
#' @references
#'  Column naming in factor encoding:
#'  mlr3pipelines::PipeOpEncode \url{https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_encode.html}
#'
#' @import mlr3pipelines
#' @export
TunerSeqMBsPLS = R6::R6Class(
  "TunerSeqMBsPLS",
  inherit = mlr3tuning::Tuner,

  public = list(

    #' @description Construct a new \code{TunerSeqMBsPLS}.
    #'
    #' @param tuner (`character(1)`)
    #'   ID of a *synchronous* mlr3 tuner for the per-component search
    #'   (e.g. `"random_search"`, `"grid_search"`).
    #' @param budget (`integer(1)`)
    #'   Maximum number of candidate evaluations **per component**.
    #' @param resampling ([\code{mlr3::Resampling}])
    #'   Inner resampling strategy (default `rsmp("cv", folds = 3)`).
    #' @param parallel (`character(1)`)
    #'   `"none"` (default) or `"inner"` to parallelise CV folds via **future**.
    #' @param early_stopping (`logical(1)`)
    #'   If \code{TRUE} (default) run a permutation test after each LC and stop
    #'   when \code{p > perm_alpha}. LC1 is always kept.
    #' @param n_perm (`integer(1)`)
    #'   Number of permutations for the test (default 1000).
    #' @param perm_alpha (`numeric(1)`)
    #'   Significance level for early stopping (default 0.05).
    #' @param performance_metric (`character(1)`)
    #'   Objective used *inside* the inner CV: `"mac"` (mean absolute correlation,
    #'   default) or `"frobenius"` (Frobenius norm of the block-score
    #'   correlation matrix).
    initialize = function(tuner          = "random_search",
                          budget         = 100L,
                          resampling     = rsmp("cv", folds = 3),
                          parallel       = "none",
                          early_stopping = TRUE,
                          n_perm         = 1000L,
                          perm_alpha     = 0.05,
                          performance_metric = "mac") {

      checkmate::assert_choice(parallel, c("none", "inner"))
      checkmate::assert_int   (budget, lower = 1L)
      checkmate::assert_flag  (early_stopping)
      checkmate::assert_int   (n_perm,  lower = 1L)
      checkmate::assert_number(perm_alpha, lower = 0, upper = 1)
      checkmate::assert_choice(performance_metric, c("mac", "frobenius"))

      if (grepl("async", tuner, ignore.case = TRUE)) {
        warning("Asynchronous tuners not supported – switching to 'random_search'")
        tuner <- "random_search"
      }

      private$.tuner          <- tuner
      private$.budget         <- budget
      private$.resampling_tpl <- resampling
      private$.parallel       <- parallel
      private$.early_stop     <- early_stopping
      private$.n_perm         <- n_perm
      private$.perm_alpha     <- perm_alpha
      private$.perf_metric    <- performance_metric

      super$initialize(
        param_set = paradox::ps(
          "performance_metric" = paradox::p_fct(
            levels  = c("mac", "frobenius"),
            default = performance_metric,
            tags    = c("train", "tune")
          )
        ),
        properties    = "single-crit",
        param_classes = c("ParamDbl", "ParamInt", "ParamFct", "ParamLgl")
      )
    },

    #' @description
    #' Run the sequential loop and populate the supplied
    #' \code{TuningInstanceSingleCrit}. The optimal block-by-component
    #' sparsity matrix is available afterwards as
    #' \code{$result$learner_param_vals$c_matrix}.
    #'
    #' @param instance ([\code{mlr3tuning::TuningInstanceSingleCrit}])
    optimize = function(instance) private$.run(instance)
  ),

  private = list(

    .tuner          = NULL,
    .budget         = NULL,
    .resampling_tpl = NULL,
    .parallel       = NULL,
    .early_stop     = NULL,
    .n_perm         = NULL,
    .perm_alpha     = NULL,
    .perf_metric    = NULL,

    # ─────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────
    .pre_graph_before_mbspls = function(learner) {
      ids <- learner$graph$ids()
      pos <- match("mbspls", ids)
      if (is.na(pos) || pos == 1L) return(mlr3pipelines::Graph$new())
      new_graph <- learner$graph$pipeops[[1]]$clone(deep = TRUE)
      if (pos == 2L) {
        return(new_graph)  # no PipeOp before MBsPLS
      } else if (pos > 2L) {
        for (i in 2:(pos - 1L)) {
          new_graph = new_graph %>>% learner$graph$pipeops[[i]]$clone(deep = TRUE)
        }
      }
      new_graph
    },

    .make_blocks = function(data, block_map, allow_encoded = TRUE) {
      cols_data <- names(data)
      esc <- function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\1", s)

      expand_cols <- function(cols) {
        unique(unlist(lapply(cols, function(cn) {
          if (cn %in% cols_data) {
            cn
          } else if (allow_encoded) {
            # capture "[base].<level>" (treatment/one-hot) and "base" if it still exists
            grep(paste0("^", esc(cn), "(\\.|$)"), cols_data, value = TRUE)
          } else {
            character(0)
          }
        }), use.names = FALSE))
      }

      lapply(block_map, function(cols) {
        ex <- expand_cols(cols)
        if (!length(ex)) {
          stop(sprintf(
            "After preprocessing, no columns matched any of: %s",
            paste(cols, collapse = ", ")
          ))
        }
        as.matrix(data[, ..ex])
      })
    },

    .deflate_blocks = function(X_blocks, W_list) {
      B <- length(X_blocks)
      for (b in seq_len(B)) {
        t_b <- X_blocks[[b]] %*% W_list[[b]]
        n   <- drop(crossprod(t_b))
        if (n > 1e-10) {
          p_b <- crossprod(X_blocks[[b]], t_b) / n
          X_blocks[[b]] <- X_blocks[[b]] - tcrossprod(t_b, p_b)
        }
      }
      X_blocks
    },

    .deflate_blocks_oos = function(X_train, X_test, W_list) {
      B <- length(X_train)
      for (b in seq_len(B)) {
        t_tr  <- X_train[[b]] %*% W_list[[b]]
        n_tr  <- drop(crossprod(t_tr))
        if (n_tr > 1e-10) {
          p_tr <- crossprod(X_train[[b]], t_tr) / n_tr
          t_te <- X_test [[b]] %*% W_list[[b]]
          X_test[[b]] <- X_test[[b]] - tcrossprod(t_te, p_tr)
        }
      }
      X_test
    },

    # ─────────────────────────────────────────────────────────────────
    #  Main optimisation loop
    # ─────────────────────────────────────────────────────────────────
    .run = function(inst) {

      learner_tpl <- inst$objective$learner
      task_full   <- inst$objective$task$clone(deep = TRUE)

      # preprocessing graph up to (but excluding) mbspls
      pre_graph_tpl <- private$.pre_graph_before_mbspls(learner_tpl)

      # correlation method
      use_spear <- tryCatch({
        identical(learner_tpl$graph$pipeops$mbspls$param_set$values$correlation_method, "spearman")
      }, error = function(e) FALSE)

      # blocks + #components from the learner
      blocks <- learner_tpl$graph$pipeops$mbspls$blocks
      K_max  <- learner_tpl$graph$pipeops$mbspls$param_set$values$ncomp %||% 1L
      B      <- length(blocks)
      if (B == 0) stop("No blocks specified in PipeOpMBsPLS")

      # preprocess FULL data once (for the final refits/deflation chain)
      pre_graph_full <- pre_graph_tpl$clone(deep = TRUE)

      if (length(pre_graph_full$pipeops)) {
        pre_graph_full$train(task_full)
        pre_df_full <- data.table::last(pre_graph_full$predict(task_full))$data()
      } else {
        pre_df_full <- task_full$data()
      }

      # --- expand block map to *preprocessed* names once -----------------
      blocks_pp <- lapply(blocks, function(cols) {
        cols_data <- names(pre_df_full)
        esc <- function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\1", s)
        unique(unlist(lapply(cols, function(cn) {
          if (cn %in% cols_data) cn else grep(paste0("^", esc(cn), "(\\.|$)"),
                                              cols_data, value = TRUE)
        })))
      })
      # overwrite for the remainder of the run (preserves block names)
      blocks <- blocks_pp

      # build the residual matrices with the normalized names
      X_blocks_residual <- private$.make_blocks(pre_df_full, blocks, allow_encoded = FALSE)
      names(X_blocks_residual) <- names(blocks)

      # instantiate inner resampling
      rs <- private$.resampling_tpl$clone()
      if (!rs$is_instantiated) rs$instantiate(task_full)
      tr_idx <- lapply(seq_len(rs$iters), rs$train_set)
      va_idx <- lapply(seq_len(rs$iters), rs$test_set)

      fold_tr  <- vector("list", rs$iters)
      fold_val <- vector("list", rs$iters)
      for (f in seq_len(rs$iters)) {
        task_tr <- task_full$clone(deep = FALSE)$filter(tr_idx[[f]])
        task_va <- task_full$clone(deep = FALSE)$filter(va_idx[[f]])

        g <- pre_graph_tpl$clone(deep = TRUE)
        tr_out <- g$train(task_tr)[[1L]]; df_tr <- tr_out$data()
        va_out <- g$predict(task_va)[[1L]]; df_va <- va_out$data()

        fold_tr[[f]]  <- private$.make_blocks(df_tr, blocks, allow_encoded = FALSE)
        fold_val[[f]] <- private$.make_blocks(df_va, blocks)
      }

      if (private$.parallel == "inner") {
        future::plan("multisession", workers = max(1L, future::availableCores() - 1L))
        on.exit(future::plan("sequential"), add = TRUE)
        fold_apply <- function(X, FUN) future.apply::future_sapply(X, FUN, future.seed = TRUE)
      } else {
        fold_apply <- function(X, FUN) sapply(X, FUN)
      }

      C_star <- matrix(NA_real_, B, K_max,
                       dimnames = list(names(blocks), paste0("LC", seq_len(K_max))))
      
      pvals_combined <- numeric(K_max)

      for (k in seq_len(K_max)) {
        lgr$info("⏩  Component %d / %d", k, K_max)

        ps_k <- do.call(
          paradox::ps,
          setNames(lapply(names(blocks), function(bn)
            paradox::p_int(lower = 1L, upper = ncol(X_blocks_residual[[bn]]))
          ),
          paste0("c_", names(blocks)))
        )

        obj_env <- new.env(parent = emptyenv())
        obj_fun <- bbotk::ObjectiveRFun$new(
          fun = function(xs) {
            key <- paste0(unlist(xs, use.names = FALSE), collapse = "_")
            if (exists(key, envir = obj_env, inherits = FALSE))
              return(list(Score = obj_env[[key]]))

            c_vec <- sqrt(unlist(xs, use.names = FALSE))

            fold_scores <- fold_apply(seq_len(rs$iters), function(f) {
              Xtr <- fold_tr[[f]]
              Xva <- fold_val[[f]]

              Wfit <- cpp_mbspls_one_lv(
                Xtr, c_vec, 1000L, 1e-4,
                frobenius = (private$.perf_metric == "frobenius")
              )$W

              B_ <- length(Xva)
              Tva <- vapply(seq_len(B_),
                            \(b) Xva[[b]] %*% as.numeric(Wfit[[b]]),
                            numeric(nrow(Xva[[1]])))

              if (B_ < 2) return(0)

              pairs <- utils::combn(seq_len(B_), 2)
              corrs <- vapply(seq_len(ncol(pairs)), function(i) {
                t1 <- Tva[, pairs[1, i]]; t2 <- Tva[, pairs[2, i]]
                if (all(abs(t1) < 1e-12) || all(abs(t2) < 1e-12)) return(0)
                stats::cor(t1, t2, method = if (use_spear) "spearman" else "pearson")
              }, numeric(1))

              if (private$.perf_metric == "mac")
                mean(abs(corrs), na.rm = TRUE)
              else
                sqrt(sum(corrs^2, na.rm = TRUE))
            })
            n_va <- vapply(seq_len(rs$iters), function(f) nrow(fold_val[[f]][[1]]), integer(1))
            score <- sum(fold_scores * n_va) / sum(n_va) # weighted mean
            # score <- mean(fold_scores)
            obj_env[[key]] <- score
            list(Score = score)
          },
          domain   = ps_k,
          codomain = paradox::ps(Score = paradox::p_dbl(tags = "maximize"))
        )

        inst_k <- bbotk::OptimInstanceBatchSingleCrit$new(
          objective    = obj_fun,
          search_space = ps_k,
          terminator   = bbotk::trm("evals", n_evals = private$.budget)
        )
        bbotk::opt(private$.tuner)$optimize(inst_k)

        C_star[, k] <- sqrt(unlist(inst_k$result_x_domain, use.names = FALSE))
        lgr$info("      best c-vector: %s", paste(round(C_star[, k], 4), collapse = ", "))

        fit_full <- cpp_mbspls_one_lv(
          X_blocks_residual, C_star[, k], 1000L, 1e-4,
          (private$.perf_metric == "frobenius")
        )

        # Per-fold refit for leakage-free deflation & OOS permutation
        p_folds <- numeric(rs$iters)
        w_va    <- numeric(rs$iters)

        for (f in seq_len(rs$iters)) {
          Xtr_before <- fold_tr[[f]]
          Xva_before <- fold_val[[f]]

          fit_fold_k <- cpp_mbspls_one_lv(
            Xtr_before, C_star[, k], 1000L, 1e-4,
            (private$.perf_metric == "frobenius")
          )

          # --- OOS permutation test on VALIDATION with weights fixed
          res <- cpp_perm_test_oos(
            X_test  = lapply(Xva_before, identity),
            W_trained = fit_fold_k$W,
            n_perm = private$.n_perm,
            spearman = FALSE,   # or TRUE if you support it consistently (see C++ note)
            frobenius = (private$.perf_metric == "frobenius"),
            early_stop_threshold = private$.perm_alpha,
            permute_all_blocks = TRUE
          )
          p_folds[f] <- as.numeric(res$p_value)
          w_va[f]    <- nrow(Xva_before[[1]])

          # --- Safe deflation for next component
          fold_tr[[f]]  <- private$.deflate_blocks(Xtr_before, fit_fold_k$W)
          fold_val[[f]] <- private$.deflate_blocks_oos(Xtr_before, Xva_before, fit_fold_k$W)
        }

        # Combine per-fold p-values via (weighted) Stouffer
        z <- stats::qnorm(pmax(1e-12, 1 - p_folds))
        w <- sqrt(w_va)
        z_comb <- sum(w * z) / sqrt(sum(w^2))
        p_k    <- 1 - stats::pnorm(z_comb)

        p_adj_prev <- if (k == 1L) 0 else max(pvals_combined[seq_len(k - 1L)])
        p_adj_k    <- max(p_adj_prev, p_k)


        if (private$.early_stop && p_adj_k > private$.perm_alpha) {
          lgr$info("      -> early stop at component %d (adj. p = %.4g)", k, p_adj_k)
          C_star <- if (k == 1L) C_star[, 1, drop = FALSE] else C_star[, seq_len(k - 1L), drop = FALSE]
          break
        } else {
          lgr$info("     Component %d significant: permutation test p = %.4g (adj. p = %.4g)", k, p_k, p_adj_k)
        }
        # keep for logging if you like
        pvals_combined[k] <- p_k

        X_blocks_residual <- private$.deflate_blocks(X_blocks_residual, fit_full$W)
        for (f in seq_len(rs$iters)) {
          fold_tr[[f]]  <- private$.deflate_blocks(fold_tr[[f]],  fit_full$W)
          fold_val[[f]] <- private$.deflate_blocks(fold_val[[f]], fit_full$W)
        }
      }

      y_name <- tryCatch(inst$objective$codomain$ids()[1], error = function(e) "score")
      minimize <- tryCatch({
        m <- inst$objective$measure %||% inst$objective$measures[[1]]
        isTRUE(m$minimize)
      }, error = function(e) FALSE)

      best_inner <- as.numeric(inst_k$result_y)
      y_val <- if (minimize) -best_inner else best_inner

      inst$assign_result(
        xdt  = data.table::data.table(),
        y    = stats::setNames(y_val, y_name),
        learner_param_vals = list(c_matrix = C_star)
      )

      invisible(inst)
    }
  )
)
