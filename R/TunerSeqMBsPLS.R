#' @title Sequential Component-wise Tuner for MB-sPLS
#'
#' @description
#' `TunerSeqMBsPLS` performs sequential hyper-parameter optimisation of a
#' multi-block sparse PLS (MB-sPLS) model in the **mlr3** ecosystem. For each
#' latent component it samples block-wise sparsity vectors `c`, scores them via
#' inner resampling using one of the package's MB-sPLS measures, optionally
#' applies a permutation-based early-stop test, and then deflates before moving
#' to the next component.
#'
#' The tuner supports the package-native MB-sPLS measures:
#' `mbspls.mac_evwt`, `mbspls.mac`, `mbspls.ev`, and `mbspls.block_ev`.
#' Passing any other measure now errors explicitly instead of being silently
#' ignored.
#'
#' @section Construction:
#' ```
#' tuner <- TunerSeqMBsPLS$new(
#'   tuner              = "random_search",
#'   budget             = 1500L,
#'   resampling         = rsmp("cv", folds = 3),
#'   parallel           = "none",
#'   early_stopping     = TRUE,
#'   n_perm             = 1000L,
#'   perm_alpha         = 0.05,
#'   performance_metric = "mac",
#'   additional_task    = NULL
#' )
#' ```
#'
#' @param tuner (`character(1)`) ID of a synchronous mlr3 tuner used for the
#'   per-component search.
#' @param budget (`integer(1)`) Maximum number of candidate evaluations per
#'   component.
#' @param resampling (`mlr3::Resampling`) Inner resampling strategy.
#' @param parallel (`character(1)`) `"none"` (default) or `"inner"`.
#' @param early_stopping (`logical(1)`) If `TRUE`, run a permutation test after
#'   each latent component and stop when `p > perm_alpha`. LC1 is always kept.
#' @param n_perm (`integer(1)`) Number of permutations for early stopping.
#' @param perm_alpha (`numeric(1)`) Significance level for early stopping.
#' @param performance_metric (`character(1)`) Correlation objective used inside
#'   `PipeOpMBsPLS`: `"mac"` or `"frobenius"`.
#' @param additional_task [mlr3::Task] or `NULL`. Optional unlabeled task whose
#'   rows are appended to the inner-CV training features when extracting
#'   MB-sPLS weights. Only features belonging to the supplied blocks are used.
#'
#' @import mlr3pipelines
#' @export
TunerSeqMBsPLS = R6::R6Class(
  "TunerSeqMBsPLS",
  inherit = mlr3tuning::Tuner,

  public = list(

    #' @description Construct a new `TunerSeqMBsPLS`.
    initialize = function(
      tuner = "random_search",
      budget = 100L,
      resampling = rsmp("cv", folds = 3),
      parallel = "none",
      early_stopping = TRUE,
      n_perm = 1000L,
      perm_alpha = 0.05,
      performance_metric = "mac",
      additional_task = NULL
    ) {

      checkmate::assert_choice(parallel, c("none", "inner"))
      checkmate::assert_int(budget, lower = 1L)
      checkmate::assert_flag(early_stopping)
      checkmate::assert_int(n_perm, lower = 1L)
      checkmate::assert_number(perm_alpha, lower = 0, upper = 1)
      checkmate::assert_choice(performance_metric, c("mac", "frobenius"))
      if (!is.null(additional_task)) {
        checkmate::assert_class(additional_task, "Task")
      }

      if (grepl("async", tuner, ignore.case = TRUE)) {
        warning("Asynchronous tuners are unsupported - switching to 'random_search'.", call. = FALSE)
        tuner = "random_search"
      }

      private$.tuner = tuner
      private$.budget = budget
      private$.resampling_tpl = resampling
      private$.parallel = parallel
      private$.early_stop = early_stopping
      private$.n_perm = n_perm
      private$.perm_alpha = perm_alpha
      private$.perf_metric = performance_metric
      private$.additional_task = additional_task

      super$initialize(
        param_set = paradox::ps(
          performance_metric = paradox::p_fct(
            levels  = c("mac", "frobenius"),
            default = performance_metric,
            tags    = c("train", "tune")
          )
        ),
        properties = "single-crit",
        param_classes = c("ParamDbl", "ParamInt", "ParamFct", "ParamLgl")
      )
    },

    #' @description Set the optional additional task after construction.
    #' @param task (`mlr3::Task`) Additional task to be appended during
    #'   component fitting.
    set_additional_task = function(task) {
      checkmate::assert_class(task, "Task")
      private$.additional_task = task
      invisible(self)
    },

    #' @description Run the sequential tuning loop.
    #' @param instance (`mlr3tuning::TuningInstanceBatchSingleCrit`) Tuning instance.
    optimize = function(instance) private$.run(instance)
  ),

  private = list(

    .tuner = NULL,
    .budget = NULL,
    .resampling_tpl = NULL,
    .parallel = NULL,
    .early_stop = NULL,
    .n_perm = NULL,
    .perm_alpha = NULL,
    .perf_metric = NULL,
    .additional_task = NULL,

    .resolve_measure = function(inst) {
      measure = tryCatch(inst$objective$measure, error = function(e) NULL)
      if (is.null(measure)) {
        measure = tryCatch(inst$objective$measures[[1L]], error = function(e) NULL)
      }
      if (is.null(measure)) {
        stop("TunerSeqMBsPLS requires a single MB-sPLS measure.", call. = FALSE)
      }
      key = .mbspls_measure_key(measure)
      if (is.null(key)) {
        stop(
          "TunerSeqMBsPLS supports only MB-sPLS measures: mbspls.mac_evwt, mbspls.mac, mbspls.ev, mbspls.block_ev.",
          call. = FALSE
        )
      }
      list(measure = measure, key = key)
    },

    .aggregate_scores = function(scores, measure, n_obs = NULL) {
      scores = as.numeric(scores)
      if (!length(scores)) {
        return(NA_real_)
      }
      avg = measure$average %||% "macro"
      if (identical(avg, "macro_weighted")) {
        w = as.numeric(n_obs %||% rep(1, length(scores)))
        return(stats::weighted.mean(scores, w = w, na.rm = FALSE))
      }
      if (identical(avg, "custom")) {
        stop("TunerSeqMBsPLS does not support custom measure aggregators.", call. = FALSE)
      }
      aggr = measure$aggregator %||% function(x) mean(x)
      aggr(scores)
    },

    .empty_fold_payload = function(block_names) {
      list(
        mac_comp = numeric(),
        ev_comp = numeric(),
        ev_block = matrix(numeric(), nrow = 0L, ncol = length(block_names),
          dimnames = list(NULL, block_names)),
        blocks = block_names,
        perf_metric = private$.perf_metric
      )
    },

    .append_fold_payload = function(payload, payload_k) {
      payload$mac_comp = c(as.numeric(payload$mac_comp), as.numeric(payload_k$mac_comp))
      payload$ev_comp = c(as.numeric(payload$ev_comp), as.numeric(payload_k$ev_comp))
      payload$ev_block = rbind(payload$ev_block, as.matrix(payload_k$ev_block))
      payload
    },

    .trim_last_component = function(payload) {
      if (length(payload$mac_comp)) {
        payload$mac_comp = payload$mac_comp[-length(payload$mac_comp)]
      }
      if (length(payload$ev_comp)) {
        payload$ev_comp = payload$ev_comp[-length(payload$ev_comp)]
      }
      if (!is.null(payload$ev_block) && nrow(payload$ev_block) > 0L) {
        payload$ev_block = payload$ev_block[seq_len(nrow(payload$ev_block) - 1L), , drop = FALSE]
      }
      payload
    },

    .pre_graph_before_mbspls = function(learner, mbspls_id = NULL) {
      ids = learner$graph$ids()
      mbspls_id = mbspls_id %||% .mbspls_pipeop_id(learner$graph, where = "learner$graph")
      pos = match(mbspls_id, ids)
      if (is.na(pos) || pos == 1L) {
        return(mlr3pipelines::Graph$new())
      }
      new_graph = learner$graph$pipeops[[1L]]$clone(deep = TRUE)
      if (pos == 2L) {
        return(new_graph)
      }
      for (i in 2:(pos - 1L)) {
        new_graph = new_graph %>>% learner$graph$pipeops[[i]]$clone(deep = TRUE)
      }
      new_graph
    },

    .make_blocks = function(data, block_map, allow_encoded = TRUE) {
      cols_data = names(data)
      esc = function(s) gsub("([][{}()|^$.*+?\\\\-])", "\\\\\\\\1", s)

      expand_cols = function(cols) {
        unique(unlist(lapply(cols, function(cn) {
          if (cn %in% cols_data) {
            cn
          } else if (allow_encoded) {
            grep(paste0("^", esc(cn), "(\\\\.|$)"), cols_data, value = TRUE)
          } else {
            character(0)
          }
        }), use.names = FALSE))
      }

      out = lapply(block_map, function(cols) {
        ex = if (allow_encoded) expand_cols(cols) else unique(cols)
        miss = setdiff(ex, names(data))
        if (length(miss)) {
          data[, (miss) := 0]
        }
        if (!length(ex)) {
          stop(sprintf(
            "After preprocessing, no columns matched any of: %s",
            paste(cols, collapse = ", ")
          ), call. = FALSE)
        }
        M = as.matrix(data[, ..ex])
        storage.mode(M) = "double"
        M
      })
      names(out) = names(block_map)
      out
    },

    .rbind_blocks = function(A, B) {
      if (is.null(B)) {
        return(A)
      }
      out = vector("list", length(A))
      for (i in seq_along(A)) {
        if (is.null(B[[i]])) {
          out[[i]] = A[[i]]
        } else {
          stopifnot(ncol(A[[i]]) == ncol(B[[i]]))
          out[[i]] = rbind(A[[i]], B[[i]])
        }
      }
      names(out) = names(A)
      out
    },

    .compute_train_loadings = function(X_blocks, W_list) {
      P_list = vector("list", length(X_blocks))
      for (b in seq_along(X_blocks)) {
        t_b = X_blocks[[b]] %*% W_list[[b]]
        denom = drop(crossprod(t_b))
        if (denom > 1e-12) {
          P_list[[b]] = as.numeric(crossprod(X_blocks[[b]], t_b) / denom)
        } else {
          P_list[[b]] = numeric(ncol(X_blocks[[b]]))
        }
      }
      names(P_list) = names(X_blocks)
      P_list
    },

    .deflate_blocks = function(X_blocks, W_list) {
      P_list = private$.compute_train_loadings(X_blocks, W_list)
      for (b in seq_along(X_blocks)) {
        tb = X_blocks[[b]] %*% W_list[[b]]
        X_blocks[[b]] = X_blocks[[b]] - tcrossprod(tb, P_list[[b]])
      }
      X_blocks
    },

    .deflate_blocks_split = function(X_tr, X_add, W_list) {
      X_fit = if (!is.null(X_add)) Map(rbind, X_tr, X_add) else X_tr
      P_fit = private$.compute_train_loadings(X_fit, W_list)

      for (b in seq_along(X_tr)) {
        tb = X_tr[[b]] %*% W_list[[b]]
        X_tr[[b]] = X_tr[[b]] - tcrossprod(tb, P_fit[[b]])
      }
      if (!is.null(X_add)) {
        for (b in seq_along(X_add)) {
          tb = X_add[[b]] %*% W_list[[b]]
          X_add[[b]] = X_add[[b]] - tcrossprod(tb, P_fit[[b]])
        }
      }
      list(train = X_tr, add = X_add, p = P_fit)
    },

    .deflate_blocks_val = function(X_tr, X_add, X_val, W_list) {
      X_fit = if (!is.null(X_add)) Map(rbind, X_tr, X_add) else X_tr
      P_fit = private$.compute_train_loadings(X_fit, W_list)
      for (b in seq_along(X_val)) {
        tb = X_val[[b]] %*% W_list[[b]]
        X_val[[b]] = X_val[[b]] - tcrossprod(tb, P_fit[[b]])
      }
      X_val
    },

    .one_lv_payload = function(X_train_fit, X_test, W_list, correlation_method) {
      P_fit = private$.compute_train_loadings(X_train_fit, W_list)
      res = compute_test_ev(
        X_blocks_test = X_test,
        W_all = list(W_list),
        P_all = list(P_fit),
        deflate = TRUE,
        performance_metric = private$.perf_metric,
        correlation_method = correlation_method,
        loading_source = "train",
        clamp_ev = "none"
      )
      list(
        mac_comp = as.numeric(res$mac_comp),
        ev_comp = as.numeric(res$ev_comp),
        ev_block = as.matrix(res$ev_block),
        blocks = names(X_test),
        perf_metric = private$.perf_metric
      )
    },

    .run = function(inst) {

      spec = private$.resolve_measure(inst)
      measure = spec$measure

      learner_tpl = inst$objective$learner
      mbspls_id = .mbspls_pipeop_id(learner_tpl$graph, where = "inst$objective$learner$graph")
      mbspls_po = learner_tpl$graph$pipeops[[mbspls_id]]
      task_full = inst$objective$task$clone(deep = TRUE)

      pre_graph_tpl = private$.pre_graph_before_mbspls(learner_tpl, mbspls_id = mbspls_id)

      use_spear = tryCatch({
        identical(mbspls_po$param_set$values$correlation_method, "spearman")
      }, error = function(e) FALSE)
      correlation_method = if (use_spear) "spearman" else "pearson"

      blocks_raw = mbspls_po$blocks
      K_max = mbspls_po$param_set$values$ncomp %||% 1L
      B = length(blocks_raw)
      if (B == 0L) {
        stop("No blocks specified in PipeOpMBsPLS.", call. = FALSE)
      }

      pre_graph_full = pre_graph_tpl$clone(deep = TRUE)
      if (length(pre_graph_full$pipeops)) {
        pre_graph_full$train(task_full)
        pre_df_full = data.table::last(pre_graph_full$predict(task_full))$data()
      } else {
        pre_df_full = task_full$data()
      }

      blocks = lapply(blocks_raw, function(cols) mb_expand_block_cols(names(pre_df_full), cols))
      names(blocks) = names(blocks_raw)
      X_blocks_residual = private$.make_blocks(pre_df_full, blocks, allow_encoded = FALSE)
      names(X_blocks_residual) = names(blocks)

      if (!is.null(private$.additional_task)) {
        if (length(pre_graph_full$pipeops)) {
          df_add_full = pre_graph_full$predict(private$.additional_task)[[1L]]$data()
        } else {
          df_add_full = private$.additional_task$data()
        }
        X_add_full = private$.make_blocks(df_add_full, blocks, allow_encoded = FALSE)
        X_blocks_residual = private$.rbind_blocks(X_blocks_residual, X_add_full)
      }

      rs = private$.resampling_tpl$clone()
      if (!rs$is_instantiated) {
        rs$instantiate(task_full)
      }

      fold_tr = vector("list", rs$iters)
      fold_val = vector("list", rs$iters)
      fold_add = if (!is.null(private$.additional_task)) vector("list", rs$iters) else NULL

      for (f in seq_len(rs$iters)) {
        task_tr = task_full$clone(deep = FALSE)$filter(rs$train_set(f))
        task_va = task_full$clone(deep = FALSE)$filter(rs$test_set(f))

        g = pre_graph_tpl$clone(deep = TRUE)
        if (length(g$pipeops)) {
          df_tr = g$train(task_tr)[[1L]]$data()
          df_va = g$predict(task_va)[[1L]]$data()
          if (!is.null(fold_add)) {
            df_add = g$predict(private$.additional_task)[[1L]]$data()
          }
        } else {
          df_tr = task_tr$data()
          df_va = task_va$data()
          if (!is.null(fold_add)) {
            df_add = private$.additional_task$data()
          }
        }

        fold_tr[[f]] = private$.make_blocks(df_tr, blocks, allow_encoded = FALSE)
        fold_val[[f]] = private$.make_blocks(df_va, blocks, allow_encoded = FALSE)
        if (!is.null(fold_add)) {
          fold_add[[f]] = private$.make_blocks(df_add, blocks, allow_encoded = FALSE)
        }
      }

      if (private$.parallel == "inner") {
        future::plan("multisession", workers = max(1L, future::availableCores() - 1L))
        on.exit(future::plan("sequential"), add = TRUE)
        fold_apply = function(X, FUN) future.apply::future_sapply(X, FUN, future.seed = TRUE)
      } else {
        fold_apply = function(X, FUN) sapply(X, FUN)
      }

      C_star = matrix(
        NA_real_, B, K_max,
        dimnames = list(names(blocks), paste0("LC", seq_len(K_max)))
      )
      pvals_combined = rep(NA_real_, K_max)
      fold_payloads = lapply(seq_len(rs$iters), function(i) private$.empty_fold_payload(names(blocks)))
      n_val = vapply(seq_len(rs$iters), function(f) nrow(fold_val[[f]][[1L]]), integer(1L))

      for (k in seq_len(K_max)) {
        lgr$info("-> MB-sPLS component %d / %d", k, K_max)

        ps_k = do.call(
          paradox::ps,
          setNames(lapply(names(blocks), function(bn) {
            paradox::p_int(lower = 1L, upper = ncol(X_blocks_residual[[bn]]))
          }), paste0("c_", names(blocks)))
        )

        obj_env = new.env(parent = emptyenv())
        obj_fun = bbotk::ObjectiveRFun$new(
          fun = function(xs) {
            key = paste0(unlist(xs, use.names = FALSE), collapse = "_")
            if (exists(key, envir = obj_env, inherits = FALSE)) {
              return(list(Score = obj_env[[key]]))
            }

            c_vec = sqrt(unlist(xs, use.names = FALSE))
            fold_scores = fold_apply(seq_len(rs$iters), function(f) {
              Xtr = fold_tr[[f]]
              Xva = fold_val[[f]]
              Xad = if (!is.null(fold_add)) fold_add[[f]] else NULL
              Xfit = private$.rbind_blocks(Xtr, Xad)
              fit = cpp_mbspls_one_lv(
                Xfit,
                c_vec,
                1000L,
                1e-4,
                frobenius = identical(private$.perf_metric, "frobenius"),
                spearman = use_spear
              )
              payload = private$.one_lv_payload(Xfit, Xva, fit$W, correlation_method = correlation_method)
              mbspls_measure_score_from_payload(payload, measure)
            })
            score_raw = private$.aggregate_scores(fold_scores, measure, n_obs = n_val)
            score_opt = if (isTRUE(measure$minimize)) -score_raw else score_raw
            obj_env[[key]] = score_opt
            list(Score = score_opt)
          },
          domain = ps_k,
          codomain = paradox::ps(Score = paradox::p_dbl(tags = "maximize"))
        )

        inst_k = bbotk::OptimInstanceBatchSingleCrit$new(
          objective = obj_fun,
          search_space = ps_k,
          terminator = bbotk::trm("evals", n_evals = private$.budget)
        )
        bbotk::opt(private$.tuner)$optimize(inst_k)

        C_star[, k] = sqrt(unlist(inst_k$result_x_domain, use.names = FALSE))
        lgr$info("   chosen c-vector: %s", paste(round(C_star[, k], 4), collapse = ", "))

        p_folds = rep(NA_real_, rs$iters)
        w_va = vapply(seq_len(rs$iters), function(f) nrow(fold_val[[f]][[1L]]), numeric(1L))

        for (f in seq_len(rs$iters)) {
          Xtr_before = fold_tr[[f]]
          Xva_before = fold_val[[f]]
          Xad_before = if (!is.null(fold_add)) fold_add[[f]] else NULL

          Xfit_before = private$.rbind_blocks(Xtr_before, Xad_before)
          fit_fold_k = cpp_mbspls_one_lv(
            Xfit_before,
            C_star[, k],
            1000L,
            1e-4,
            frobenius = identical(private$.perf_metric, "frobenius"),
            spearman = use_spear
          )

          payload_k = private$.one_lv_payload(Xfit_before, Xva_before, fit_fold_k$W, correlation_method = correlation_method)
          fold_payloads[[f]] = private$.append_fold_payload(fold_payloads[[f]], payload_k)

          if (private$.early_stop) {
            res = try(
              cpp_perm_test_oos(
                X_test = lapply(Xva_before, identity),
                W_trained = fit_fold_k$W,
                n_perm = private$.n_perm,
                spearman = use_spear,
                frobenius = identical(private$.perf_metric, "frobenius"),
                early_stop_threshold = private$.perm_alpha,
                permute_all_blocks = TRUE
              ),
              silent = TRUE
            )
            p_folds[f] = if (inherits(res, "try-error")) 1 else as.numeric(res$p_value)
          }

          spl = private$.deflate_blocks_split(Xtr_before, Xad_before, fit_fold_k$W)
          fold_tr[[f]] = spl$train
          if (!is.null(fold_add)) {
            fold_add[[f]] = spl$add
          }
          fold_val[[f]] = private$.deflate_blocks_val(Xtr_before, Xad_before, Xva_before, fit_fold_k$W)
        }

        if (private$.early_stop) {
          z = stats::qnorm(pmax(1e-12, 1 - p_folds))
          w = sqrt(w_va)
          z_comb = sum(w * z) / sqrt(sum(w^2))
          p_k = 1 - stats::pnorm(z_comb)
          p_adj_prev = if (k == 1L) 0 else max(pvals_combined[seq_len(k - 1L)], na.rm = TRUE)
          if (!is.finite(p_adj_prev)) p_adj_prev = 0
          p_adj_k = max(p_adj_prev, p_k)
          pvals_combined[k] = p_k

          if (p_adj_k > private$.perm_alpha) {
            lgr$info("   early stop at component %d (adj. p = %.4g)", k, p_adj_k)
            if (k > 1L) {
              C_star = C_star[, seq_len(k - 1L), drop = FALSE]
              fold_payloads = lapply(fold_payloads, private$.trim_last_component)
            } else {
              C_star = C_star[, 1L, drop = FALSE]
            }
            break
          }
          lgr$info("   component %d significant: permutation p = %.4g (adj. p = %.4g)", k, p_k, p_adj_k)
        }

        fit_full = cpp_mbspls_one_lv(
          X_blocks_residual,
          C_star[, k],
          1000L,
          1e-4,
          frobenius = identical(private$.perf_metric, "frobenius"),
          spearman = use_spear
        )
        X_blocks_residual = private$.deflate_blocks(X_blocks_residual, fit_full$W)
      }

      fold_scores_final = vapply(fold_payloads, function(pl) {
        mbspls_measure_score_from_payload(pl, measure)
      }, numeric(1L))
      y_raw = private$.aggregate_scores(fold_scores_final, measure, n_obs = n_val)

      inst$assign_result(
        xdt = data.table::data.table(),
        y = stats::setNames(y_raw, measure$id),
        learner_param_vals = list(c_matrix = C_star)
      )

      invisible(inst)
    }
  )
)
