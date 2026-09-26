#' @title Sequential Component-wise Tuner for MB-sPLS
#'
#' @description
#' `TunerSeqMBsPLS` performs sequential hyper-parameter optimisation of a
#' multi-block sparse PLS (MB-sPLS) model in the **mlr3** ecosystem. For each
#' latent component it samples block-wise sparsity vectors `c`, scores them via
#' inner resampling using one of the package's MB-sPLS measures, optionally
#' applies a permutation-based early-stopping heuristic, and then deflates
#' before moving to the next component.
#'
#' The tuner supports the package-native MB-sPLS measures:
#' `mbspls.mac_evwt`, `mbspls.mac`, `mbspls.ev`, and `mbspls.block_ev`.
#' Passing any other measure errors explicitly instead of being silently
#' ignored.
#'
#' @details
#' **Inner folds mirror the final fit.** For every inner fold, the
#' preprocessing graph upstream of `PipeOpMBsPLS` is trained on the fold's
#' training rows only and then applied to its validation rows and to
#' `additional_task`. Block columns are resolved exactly as `PipeOpMBsPLS`
#' resolves them, from the feature columns it receives (never the target, and
#' only those selected by its `affect_columns`): [mb_resolve_block_columns()]
#' (so encoded factor columns such as `sex.m` belong to their declared block),
#' followed by the numeric and finite-variance filters. Blocks without a usable
#' column on the full tuning task are dropped with a warning, as
#' `PipeOpMBsPLS` drops them, and the tuned `c_matrix` has one row per retained
#' block. As in `PipeOpMBsPLS`, the node's `ncomp` may not exceed the rank of
#' any centred retained block on the full tuning task; this is checked before
#' the search. Every retained block must keep at least one usable column in
#' each inner training fold. The upper bound of a block's `c` is `sqrt(p)` for
#' the smallest width `p` of that block on the full task and across the inner
#' training folds. An error raised while scoring a candidate, for example by
#' the solver, stops the tuning with its original message.
#'
#' The native solver expects column-centred blocks. Each inner fold is centred
#' by the column means of the rows its components are fitted on (the fold's
#' training rows plus any `additional_task` rows), and the validation rows are
#' centred by the same means, as `PipeOpMBsPLS` centres prediction data by its
#' training means.
#'
#' Fold fits use the solver settings of `PipeOpMBsPLS`: at most 600 sweeps,
#' stopping when the largest change in block weights between sweeps falls below
#' `1e-4`. The solver starts deterministically, so the fit that scored the
#' selected candidate is reused for the fold payloads, the early-stopping
#' diagnostic and the deflation, and it equals the fit `PipeOpMBsPLS` obtains
#' for the same data and `c`. A warning names retained components whose fold
#' fits did not converge.
#'
#' **Reported score.** `result_y` is the inner-resampling score of the tuned
#' `c_matrix`. Each component's `c` is chosen to maximise this score on the same
#' inner folds, so it is optimistically biased (for a single component it is the
#' best score in the search archive). Use [mbspls_nested_cv()] to estimate
#' performance on unseen data.
#'
#' **Early stopping is a heuristic.** After the `c` of component `k` is
#' selected, a permutation test that permutes the validation rows with the
#' trained weights held fixed is run on every inner validation fold. These are
#' the folds that selected `c`, and the fixed-weight null ignores that
#' selection, so the fold p-values are optimistic (anti-conservative), the more
#' so the larger `budget`. The fold p-values are pooled with a
#' `sqrt(n_validation)`-weighted Stouffer combination that treats the folds as
#' independent, although their training rows overlap. Tuning stops at the first
#' component whose running maximum of pooled p-values exceeds `perm_alpha`, and
#' that component is dropped; LC1 is always kept. `perm_alpha` is therefore a
#' cutoff for this heuristic, not a false-retention or type-I error rate, and
#' the pooled values are not confirmatory p-values.
#'
#' The performance metric can also be changed after construction through
#' `tuner$param_set$values$performance_metric`; the parameter set is the only
#' source of this setting.
#'
#' @section Construction:
#' ```
#' tuner = TunerSeqMBsPLS$new(
#'   tuner              = "random_search",
#'   budget             = 100L,
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
#' @param parallel (`character(1)`) `"none"` (default) or `"inner"`, which
#'   fits the inner folds of each candidate in parallel with `future`
#'   (multisession). Fold fits draw no random numbers, so both settings propose
#'   the same candidates for a given seed.
#' @param early_stopping (`logical(1)`) If `TRUE`, run the permutation-based
#'   stopping heuristic after each latent component (see Details) and stop
#'   when its sequential pooled p-value exceeds `perm_alpha`. LC1 is always
#'   kept. The diagnostic reuses the inner validation folds that selected `c`
#'   and is optimistic; it is not confirmatory inference.
#' @param n_perm (`integer(1)`) Number of permutations for early stopping.
#' @param perm_alpha (`numeric(1)`) Cutoff of the early-stopping heuristic. It
#'   is not an error rate: the pooled fold p-values ignore the selection of `c`
#'   on the same folds and the dependence between folds.
#' @param performance_metric (`character(1)`) Correlation objective used inside
#'   `PipeOpMBsPLS`: `"mac"` or `"frobenius"`. Initial value of the parameter
#'   `performance_metric`; it must match the `PipeOpMBsPLS` node.
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
        stop(
          "Asynchronous tuners are unsupported by TunerSeqMBsPLS because the sequential component path depends on deterministic synchronous evaluations. Choose a synchronous tuner such as 'random_search'.",
          call. = FALSE
        )
      }

      private$.tuner = tuner
      private$.budget = budget
      private$.resampling_tpl = resampling
      private$.parallel = parallel
      private$.early_stop = early_stopping
      private$.n_perm = n_perm
      private$.perm_alpha = perm_alpha
      private$.additional_task = additional_task

      super$initialize(
        param_set = paradox::ps(
          performance_metric = paradox::p_fct(
            levels = c("mac", "frobenius"),
            tags = "required"
          )
        ),
        properties = "single-crit",
        param_classes = c("ParamDbl", "ParamInt", "ParamFct", "ParamLgl")
      )
      self$param_set$values$performance_metric = performance_metric
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

  active = list(
    #' @field diagnostics (`list()` or `NULL`)\cr
    #' Diagnostics of the last completed `optimize()` call: `blocks`, the
    #' resolved block columns on the full tuning task; `fold_statistics`, a
    #' [data.table::data.table()] with one row per evaluated component and inner
    #' fold (`component`, `fold`, `n_train` (fitted rows, including
    #' `additional_task` rows), `n_validation`, the fold `score` of the
    #' selected `c`, solver `converged` and `iterations`, the fold
    #' early-stopping `p_value`, and `kept`); and `components`, one row per
    #' evaluated component with its `c_<block>` values, `search_score` (score of
    #' the selected candidate in the search archive), `n_evals`, the pooled
    #' early-stopping `p_combined` and its running maximum `p_sequential`, and
    #' `kept`. Read-only.
    diagnostics = function(rhs) {
      if (!missing(rhs)) {
        stop("`diagnostics` is read-only.", call. = FALSE)
      }
      private$.diagnostics
    }
  ),

  private = list(

    .tuner = NULL,
    .budget = NULL,
    .resampling_tpl = NULL,
    .parallel = NULL,
    .early_stop = NULL,
    .n_perm = NULL,
    .perm_alpha = NULL,
    .additional_task = NULL,
    .diagnostics = NULL,

    .performance_metric = function() {
      pm = self$param_set$values$performance_metric
      checkmate::assert_choice(pm, c("mac", "frobenius"), .var.name = "performance_metric")
      pm
    },

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

    .score_payload = function(payload, measure) {
      mbspls_measure_score_diagnostics(payload, measure)
    },

    .empty_fold_payload = function(block_names, performance_metric = private$.performance_metric()) {
      list(
        mac_comp = numeric(),
        ev_comp = numeric(),
        ev_block = matrix(numeric(), nrow = 0L, ncol = length(block_names),
          dimnames = list(NULL, block_names)),
        blocks = block_names,
        perf_metric = performance_metric
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
      mbspls_id = mbspls_id %||% .mbspls_pipeop_id(learner$graph, where = "learner$graph")
      mb_preprocessing_graph(learner$graph, mbspls_id)
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
          if (ncol(A[[i]]) != ncol(B[[i]])) {
            stop(sprintf(
              "Cannot row-bind block '%s': %d training columns versus %d additional-task columns.",
              names(A)[i],
              ncol(A[[i]]),
              ncol(B[[i]])
            ), call. = FALSE)
          }
          cols_a = colnames(A[[i]])
          cols_b = colnames(B[[i]])
          if (!identical(cols_a, cols_b)) {
            if (!is.null(cols_a) && !is.null(cols_b) && setequal(cols_a, cols_b)) {
              B[[i]] = B[[i]][, cols_a, drop = FALSE]
            } else {
              stop(sprintf(
                "Cannot row-bind block '%s': column names differ between the main task and additional_task.",
                names(A)[i]
              ), call. = FALSE)
            }
          }
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

    .one_lv_payload = function(X_train_fit, X_test, W_list, correlation_method, original_ss = NULL,
      performance_metric = private$.performance_metric()) {
      P_fit = private$.compute_train_loadings(X_train_fit, W_list)
      res = compute_test_ev(
        X_blocks_test = X_test,
        W_all = list(W_list),
        P_all = list(P_fit),
        deflate = TRUE,
        performance_metric = performance_metric,
        correlation_method = correlation_method,
        loading_source = "train",
        clamp_ev = "none"
      )
      if (!is.null(original_ss)) {
        residual_ss = vapply(X_test, function(block) sum(block^2), numeric(1L))
        block_ratio = ifelse(original_ss > 1e-12, residual_ss / original_ss, 0)
        res$ev_block = sweep(as.matrix(res$ev_block), 2L, block_ratio, `*`)
        res$ev_comp = as.numeric(res$ev_comp) * if (sum(original_ss) > 1e-12) {
          sum(residual_ss) / sum(original_ss)
        } else {
          0
        }
      }
      list(
        mac_comp = as.numeric(res$mac_comp),
        ev_comp = as.numeric(res$ev_comp),
        ev_block = as.matrix(res$ev_block),
        blocks = names(X_test),
        perf_metric = performance_metric
      )
    },

    .run = function(inst) {

      spec = private$.resolve_measure(inst)
      measure = spec$measure
      perf_metric = private$.performance_metric()
      use_frob = identical(perf_metric, "frobenius")
      private$.diagnostics = NULL

      learner_tpl = inst$objective$learner
      mbspls_id = .mbspls_pipeop_id(learner_tpl$graph, where = "inst$objective$learner$graph")
      mbspls_po = learner_tpl$graph$pipeops[[mbspls_id]]
      po_vals = utils::modifyList(
        paradox::default_values(mbspls_po$param_set),
        mbspls_po$param_set$values,
        keep.null = TRUE
      )
      learner_perf = po_vals$performance_metric
      if (is.null(learner_perf) || !learner_perf %in% c("mac", "frobenius")) {
        stop("PipeOpMBsPLS must expose a valid 'performance_metric' parameter value ('mac' or 'frobenius').", call. = FALSE)
      }
      if (!identical(learner_perf, perf_metric)) {
        stop(
          sprintf(
            "TunerSeqMBsPLS performance_metric ('%s') must match the PipeOpMBsPLS node ('%s').",
            perf_metric,
            learner_perf
          ),
          call. = FALSE
        )
      }
      task_full = inst$objective$task$clone(deep = TRUE)

      pre_graph_tpl = private$.pre_graph_before_mbspls(learner_tpl, mbspls_id = mbspls_id)

      correlation_method = po_vals$correlation_method
      if (is.null(correlation_method) || !correlation_method %in% c("pearson", "spearman")) {
        stop("PipeOpMBsPLS must expose a valid 'correlation_method' parameter value ('pearson' or 'spearman').", call. = FALSE)
      }
      use_spear = identical(correlation_method, "spearman")

      # Same effective value the PipeOp fits (a param_set override wins).
      blocks_raw = po_vals$blocks %||% mbspls_po$blocks
      K_max = po_vals$ncomp %||% 1L
      if (!length(blocks_raw)) {
        stop("No blocks specified in PipeOpMBsPLS.", call. = FALSE)
      }

      # Resolve the blocks the final PipeOpMBsPLS fit will use when the tuned
      # learner is trained on the full tuning task, and apply its rank guard.
      # No model is fitted here.
      pre_graph_full = pre_graph_tpl$clone(deep = TRUE)
      if (length(pre_graph_full$pipeops)) {
        pre_df_full = .mb_tuner_feature_data(pre_graph_full$train(task_full)[[1L]], mbspls_po)
      } else {
        pre_df_full = .mb_tuner_feature_data(task_full, mbspls_po)
      }
      blocks = .mb_tuner_retained_blocks(pre_df_full, blocks_raw, "PipeOpMBsPLS")
      B = length(blocks)
      .mb_tuner_assert_rank(pre_df_full, blocks, K_max, "PipeOpMBsPLS")

      rs = private$.resampling_tpl$clone()
      if (!rs$is_instantiated) {
        rs$instantiate(task_full)
      }
      n_folds = rs$iters

      fold_tr = vector("list", n_folds)
      fold_val = vector("list", n_folds)
      fold_add = if (!is.null(private$.additional_task)) vector("list", n_folds) else NULL

      for (f in seq_len(n_folds)) {
        mb_assert_resampling_split(task_full, rs$train_set(f), rs$test_set(f))
        task_tr = task_full$clone(deep = FALSE)$filter(rs$train_set(f))
        task_va = task_full$clone(deep = FALSE)$filter(rs$test_set(f))

        g = pre_graph_tpl$clone(deep = TRUE)
        if (length(g$pipeops)) {
          df_tr = .mb_tuner_feature_data(g$train(task_tr)[[1L]], mbspls_po)
          df_va = .mb_tuner_feature_data(g$predict(task_va)[[1L]], mbspls_po)
          if (!is.null(fold_add)) {
            df_add = .mb_tuner_feature_data(g$predict(private$.additional_task)[[1L]], mbspls_po)
          }
        } else {
          df_tr = .mb_tuner_feature_data(task_tr, mbspls_po)
          df_va = .mb_tuner_feature_data(task_va, mbspls_po)
          if (!is.null(fold_add)) {
            df_add = .mb_tuner_feature_data(private$.additional_task, mbspls_po)
          }
        }

        fold_blocks = .mb_tuner_fold_blocks(df_tr, blocks_raw, names(blocks), f)
        X_tr = .mb_tuner_block_matrices(df_tr, fold_blocks, sprintf("inner training fold %d", f))
        X_va = .mb_tuner_block_matrices(df_va, fold_blocks, sprintf("inner validation fold %d", f))
        X_ad = if (!is.null(fold_add)) {
          .mb_tuner_block_matrices(df_add, fold_blocks, sprintf("additional_task in inner fold %d", f))
        }
        centred = .mb_center_fold_blocks(X_tr, X_va, X_ad)
        fold_tr[[f]] = centred$train
        fold_val[[f]] = centred$validation
        if (!is.null(fold_add)) {
          fold_add[[f]] = centred$additional
        }
      }

      fold_ss = lapply(fold_val, function(X) {
        vapply(X, function(block) sum(block^2), numeric(1L))
      })
      budget_width = vapply(names(blocks), function(block) {
        min(c(length(blocks[[block]]), vapply(fold_tr, function(X) {
          ncol(X[[block]])
        }, integer(1L))))
      }, numeric(1L))

      if (private$.parallel == "inner") {
        if (!requireNamespace("future", quietly = TRUE) || !requireNamespace("future.apply", quietly = TRUE)) {
          stop(
            "parallel = 'inner' requires the optional packages 'future' and 'future.apply'.",
            call. = FALSE
          )
        }
        old_plan = future::plan("multisession", workers = max(1L, future::availableCores() - 1L))
        on.exit(future::plan(old_plan), add = TRUE)
        # Fold fits draw no random numbers, so no RNG streams are set up and the
        # search proposes the same candidates as a sequential run.
        fold_map = function(X, FUN) future.apply::future_lapply(X, FUN, future.seed = NULL)
      } else {
        fold_map = function(X, FUN) lapply(X, FUN)
      }

      # One-LV fit of the current (deflated) fold data, its held-out payload and score.
      fit_fold = function(f, c_vec) {
        X_fit = private$.rbind_blocks(fold_tr[[f]], if (!is.null(fold_add)) fold_add[[f]])
        fit = cpp_mbspls_one_lv(
          X_fit,
          c_vec,
          600L,
          1e-4,
          frobenius = use_frob,
          spearman = use_spear
        )
        payload = private$.one_lv_payload(X_fit, fold_val[[f]], fit$W,
          correlation_method = correlation_method, original_ss = fold_ss[[f]],
          performance_metric = perf_metric)
        list(fit = fit, payload = payload, score = private$.score_payload(payload, measure))
      }

      C_star = matrix(
        NA_real_, B, K_max,
        dimnames = list(names(blocks), paste0("LC", seq_len(K_max)))
      )
      pvals_combined = rep(NA_real_, K_max)
      fold_payloads = lapply(seq_len(n_folds), function(i) {
        private$.empty_fold_payload(names(blocks), performance_metric = perf_metric)
      })
      n_val = vapply(seq_len(n_folds), function(f) nrow(fold_val[[f]][[1L]]), integer(1L))
      n_fit = vapply(seq_len(n_folds), function(f) {
        nrow(fold_tr[[f]][[1L]]) + if (!is.null(fold_add)) nrow(fold_add[[f]][[1L]]) else 0L
      }, integer(1L))
      fold_stats = list()
      component_stats = list()

      for (k in seq_len(K_max)) {
        lgr$info("-> MB-sPLS component %d / %d", k, K_max)

        ps_k = do.call(
          paradox::ps,
          setNames(lapply(names(blocks), function(bn) {
            paradox::p_int(lower = 1L, upper = budget_width[[bn]])
          }), paste0("c_", names(blocks)))
        )

        obj_env = new.env(parent = emptyenv())
        obj_diag_env = new.env(parent = emptyenv())
        # Fold fits of the best candidate so far, reused after the search, and
        # the first error raised while scoring a candidate.
        best = new.env(parent = emptyenv())
        best$key = NULL
        best$score = -Inf
        best$error = NULL
        obj_fun = bbotk::ObjectiveRFun$new(
          fun = function(xs) {
            key = paste0(unlist(xs, use.names = FALSE), collapse = "_")
            if (exists(key, envir = obj_env, inherits = FALSE)) {
              return(list(Score = obj_env[[key]]))
            }

            c_vec = sqrt(unlist(xs, use.names = FALSE))
            fold_results = tryCatch(
              fold_map(seq_len(n_folds), function(f) fit_fold(f, c_vec)),
              error = function(e) {
                best$error = e
                stop(e)
              }
            )
            fold_scores = vapply(fold_results, function(res) as.numeric(res$score$score), numeric(1L))
            failed_folds = sum(!vapply(fold_results, function(res) isTRUE(res$score$defined), logical(1L)))

            score_opt = if (!failed_folds && all(is.finite(fold_scores))) {
              score_raw = private$.aggregate_scores(fold_scores, measure, n_obs = n_val)
              if (isTRUE(measure$minimize)) -score_raw else score_raw
            } else {
              -Inf
            }
            obj_env[[key]] = score_opt
            obj_diag_env[[key]] = list(score = score_opt, failed_folds = failed_folds)
            if (is.null(best$key) || score_opt > best$score) {
              best$key = key
              best$score = score_opt
              best$folds = fold_results
            }
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
        optimize_error = tryCatch(
          {
            bbotk::opt(private$.tuner)$optimize(inst_k)
            NULL
          },
          error = function(e) e
        )

        # An error raised while scoring a candidate (for example by the solver)
        # is reported as it is. Only when every evaluated candidate scored -Inf,
        # which bbotk cannot turn into a result, is the undefined-score error
        # raised instead.
        if (!is.null(best$error)) {
          .mb_tuner_rethrow(best$error, sprintf("TunerSeqMBsPLS failed while scoring component %d", k))
        }
        diag_keys = ls(obj_diag_env, all.names = TRUE)
        diag_scores = vapply(diag_keys, function(key) as.numeric(obj_diag_env[[key]]$score), numeric(1L))
        if (length(diag_keys) && all(diag_scores %in% -Inf)) {
          failed_per_candidate = vapply(diag_keys, function(key) {
            as.integer(obj_diag_env[[key]]$failed_folds %||% 0L)
          }, integer(1L))
          stop(sprintf(
            "All %d candidate c-vectors produced an undefined '%s' score for component %d (failed folds per candidate ranged from %d to %d of %d). Increase data support for the requested objective or choose a different MB-sPLS measure for tuning.",
            length(diag_keys),
            measure$id,
            k,
            min(failed_per_candidate),
            max(failed_per_candidate),
            n_folds
          ), call. = FALSE)
        }
        if (!is.null(optimize_error)) {
          .mb_tuner_rethrow(optimize_error, sprintf("TunerSeqMBsPLS failed while tuning component %d", k))
        }
        archive_data = inst_k$archive$data

        x_star = unlist(inst_k$result_x_domain, use.names = FALSE)
        C_star[, k] = sqrt(x_star)
        lgr$info("   chosen c-vector: %s", paste(round(C_star[, k], 4), collapse = ", "))
        search_score = as.numeric(inst_k$result_y)
        if (isTRUE(measure$minimize)) {
          search_score = -search_score
        }

        # The deterministic solver reproduces the search fits of the selected
        # candidate, so they are reused rather than refitted.
        folds_k = if (identical(best$key, paste0(x_star, collapse = "_"))) {
          best$folds
        } else {
          lapply(seq_len(n_folds), function(f) fit_fold(f, C_star[, k]))
        }

        p_folds = rep(NA_real_, n_folds)
        for (f in seq_len(n_folds)) {
          X_tr_before = fold_tr[[f]]
          X_va_before = fold_val[[f]]
          X_ad_before = if (!is.null(fold_add)) fold_add[[f]] else NULL
          W_k = folds_k[[f]]$fit$W

          fold_payloads[[f]] = private$.append_fold_payload(fold_payloads[[f]], folds_k[[f]]$payload)

          if (private$.early_stop) {
            res = tryCatch(
              cpp_perm_test_oos(
                X_test = lapply(X_va_before, identity),
                W_trained = W_k,
                n_perm = private$.n_perm,
                spearman = use_spear,
                frobenius = use_frob,
                early_stop_threshold = private$.perm_alpha,
                permute_all_blocks = TRUE
              ),
              error = function(e) {
                stop(sprintf(
                  "Early-stopping permutation test failed at component %d, fold %d: %s",
                  k,
                  f,
                  conditionMessage(e)
                ), call. = FALSE)
              }
            )
            p_folds[f] = as.numeric(res$p_value %||% NA_real_)
            if (!is.finite(p_folds[f])) {
              stop(sprintf(
                "Early-stopping permutation test returned a non-finite p-value at component %d, fold %d.",
                k,
                f
              ), call. = FALSE)
            }
          }

          spl = private$.deflate_blocks_split(X_tr_before, X_ad_before, W_k)
          fold_tr[[f]] = spl$train
          if (!is.null(fold_add)) {
            fold_add[[f]] = spl$add
          }
          fold_val[[f]] = private$.deflate_blocks_val(X_tr_before, X_ad_before, X_va_before, W_k)
        }

        fold_stats[[k]] = data.table::data.table(
          component = k,
          fold = seq_len(n_folds),
          n_train = n_fit,
          n_validation = n_val,
          score = vapply(folds_k, function(res) as.numeric(res$score$score), numeric(1L)),
          converged = vapply(folds_k, function(res) as.logical(res$fit$converged %||% NA)[1L], logical(1L)),
          iterations = vapply(folds_k, function(res) as.integer(res$fit$iterations %||% NA_integer_)[1L], integer(1L)),
          p_value = p_folds
        )

        p_k = NA_real_
        p_adj_k = NA_real_
        stop_here = FALSE
        if (private$.early_stop) {
          z = stats::qnorm(pmax(1e-12, 1 - p_folds))
          w = sqrt(n_val)
          z_comb = sum(w * z) / sqrt(sum(w^2))
          p_k = 1 - stats::pnorm(z_comb)
          p_adj_prev = if (k == 1L) 0 else max(pvals_combined[seq_len(k - 1L)], na.rm = TRUE)
          if (!is.finite(p_adj_prev)) p_adj_prev = 0
          p_adj_k = max(p_adj_prev, p_k)
          pvals_combined[k] = p_k
          stop_here = p_adj_k > private$.perm_alpha
        }
        component_stats[[k]] = cbind(
          data.table::data.table(component = k),
          data.table::as.data.table(as.list(stats::setNames(C_star[, k], paste0("c_", names(blocks))))),
          data.table::data.table(
            search_score = search_score,
            n_evals = nrow(archive_data),
            p_combined = p_k,
            p_sequential = p_adj_k
          )
        )

        if (stop_here) {
          lgr$info("   early stop at component %d (adj. p = %.4g)", k, p_adj_k)
          if (k > 1L) {
            C_star = C_star[, seq_len(k - 1L), drop = FALSE]
            fold_payloads = lapply(fold_payloads, private$.trim_last_component)
          } else {
            C_star = C_star[, 1L, drop = FALSE]
          }
          break
        }
        if (private$.early_stop) {
          lgr$info("   component %d passed the early-stopping cutoff: pooled p = %.4g (sequential maximum = %.4g)", k, p_k, p_adj_k)
        }
      }

      n_kept = ncol(C_star)
      fold_statistics = data.table::rbindlist(fold_stats)
      fold_statistics$kept = fold_statistics$component <= n_kept
      components = data.table::rbindlist(component_stats)
      components$kept = components$component <= n_kept

      .mb_tuner_warn_nonconverged(
        fold_statistics,
        "TunerSeqMBsPLS: the MB-sPLS solver did not converge within 600 sweeps",
        "LC"
      )

      fold_results_final = lapply(fold_payloads, function(pl) {
        private$.score_payload(pl, measure)
      })
      fold_scores_final = vapply(fold_results_final, function(res) as.numeric(res$score), numeric(1L))
      failed_final_folds = sum(!vapply(fold_results_final, function(res) isTRUE(res$defined), logical(1L)))
      if (failed_final_folds) {
        stop(sprintf(
          "The selected c-matrix produced an undefined '%s' score in %d/%d inner validation folds. Increase data support for the requested objective or choose a different MB-sPLS measure for tuning.",
          measure$id,
          failed_final_folds,
          length(fold_results_final)
        ), call. = FALSE)
      }
      y_raw = private$.aggregate_scores(fold_scores_final, measure, n_obs = n_val)

      private$.diagnostics = list(
        blocks = blocks,
        fold_statistics = fold_statistics,
        components = components
      )
      inst$assign_result(
        xdt = data.table::data.table(),
        y = stats::setNames(y_raw, measure$id),
        learner_param_vals = list(c_matrix = C_star)
      )

      invisible(inst)
    }
  )
)

# ------------------------------------------------------------------------------
# Helpers shared by TunerSeqMBsPLS and TunerSeqMBsPCA
# ------------------------------------------------------------------------------

# Feature columns the tuned PipeOp receives from `task`: the features (never
# the target), restricted by its `affect_columns` selector when one is set.
.mb_tuner_feature_data = function(task, pipeop) {
  selector = pipeop$param_set$values$affect_columns
  cols = if (is.null(selector)) task$feature_names else selector(task)
  task$data(cols = cols)
}

# Resolve block columns as PipeOpMBsPLS, PipeOpMBsPLSXY and PipeOpMBsPCA do:
# shared resolver first, then the numeric and finite-variance filters. Blocks
# without a usable column come back empty.
.mb_tuner_resolve_blocks = function(data, block_map) {
  resolved = mb_resolve_block_columns(names(data), block_map)
  lapply(resolved, function(cols) {
    cols[vapply(cols, function(column) {
      is.numeric(data[[column]]) && mb_has_finite_variance(data[[column]])
    }, logical(1L))]
  })
}

# Blocks of the final fit on the full tuning task. Empty blocks are dropped
# with a warning, as the tuned PipeOp drops them.
.mb_tuner_retained_blocks = function(data, block_map, pipeop) {
  blocks = .mb_tuner_resolve_blocks(data, block_map)
  empty = names(blocks)[!lengths(blocks)]
  blocks = blocks[lengths(blocks) > 0L]
  if (!length(blocks)) {
    stop("No block contains at least one numeric, non-constant feature on the tuning task.", call. = FALSE)
  }
  if (length(empty)) {
    warning(sprintf(
      "Blocks without numeric, non-constant features on the tuning task are dropped, as %s drops them: %s. The tuned c_matrix has one row per retained block.",
      pipeop,
      paste(empty, collapse = ", ")
    ), call. = FALSE)
  }
  blocks
}

# The tuned PipeOp refuses more components than the rank of a centred
# training block. The same guard runs on the full tuning task before any
# search, so an infeasible `ncomp` fails with the message of the final fit.
# Returns the centred full-task blocks.
.mb_tuner_assert_rank = function(data, blocks, ncomp, pipeop) {
  X = .mb_tuner_block_matrices(data, blocks, "the full tuning task")
  X = lapply(X, function(M) sweep(M, 2L, colMeans(M), `-`))
  .mb_assert_component_rank(X, ncomp, sprintf("%s (full tuning task)", pipeop))
  X
}

# Warn once, naming the retained components and inner folds whose fold fits
# did not converge.
.mb_tuner_warn_nonconverged = function(fold_statistics, what, prefix) {
  not_converged = fold_statistics$kept & fold_statistics$converged %in% FALSE
  if (!any(not_converged)) {
    return(invisible(NULL))
  }
  failed = split(fold_statistics$fold[not_converged], fold_statistics$component[not_converged])
  warning(sprintf(
    "%s for retained component(s) %s. Their tuned sparsity and inner scores may be unreliable.",
    what,
    paste(sprintf("%s%s (inner folds %s)", prefix, names(failed),
      vapply(failed, paste, character(1L), collapse = ", ")), collapse = "; ")
  ), call. = FALSE)
}

# Re-raise a condition caught during tuning with `context` prefixed to its
# message, keeping its class.
.mb_tuner_rethrow = function(e, context) {
  e$message = sprintf("%s: %s", context, conditionMessage(e))
  e$call = NULL
  stop(e)
}

# Blocks of one inner training fold. Resolution uses the complete declared
# mapping, so a dropped block's names are never claimed by another block.
.mb_tuner_fold_blocks = function(data, block_map, retained, fold) {
  blocks = .mb_tuner_resolve_blocks(data, block_map)[retained]
  empty = names(blocks)[!lengths(blocks)]
  if (length(empty)) {
    stop(sprintf(
      "No usable training features remain in inner training fold %d for blocks: %s. These blocks are usable on the full tuning task, so the final model would include them.",
      fold,
      paste(empty, collapse = ", ")
    ), call. = FALSE)
  }
  blocks
}

.mb_tuner_block_matrices = function(data, blocks, context) {
  out = lapply(names(blocks), function(bn) {
    cols = blocks[[bn]]
    mb_assert_columns_present(
      colnames_dt = names(data),
      required = cols,
      context = sprintf("Preprocessed data (%s)", context),
      hint = "Ensure that all block columns survive upstream preprocessing in every inner fold."
    )
    M = .mb_numeric_matrix(
      as.matrix(data[, cols, with = FALSE]),
      sprintf("block '%s' in %s", bn, context)
    )
    storage.mode(M) = "double"
    M
  })
  names(out) = names(blocks)
  out
}

# Centre the blocks of one inner fold by the column means of the rows the
# components are fitted on (training rows plus any additional rows) and apply
# the same means to the validation rows, as the PipeOps centre prediction data
# by their training means.
.mb_center_fold_blocks = function(train, validation, additional = NULL) {
  fit = if (is.null(additional)) train else Map(rbind, train, additional)
  center = lapply(fit, colMeans)
  shift = function(X) {
    out = Map(function(M, m) sweep(M, 2L, m, `-`), X, center)
    names(out) = names(X)
    out
  }
  list(
    train = shift(train),
    validation = shift(validation),
    additional = if (!is.null(additional)) shift(additional),
    center = center
  )
}
