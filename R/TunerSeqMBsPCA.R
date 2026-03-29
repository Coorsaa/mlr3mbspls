#' @title Sequential Component-wise Tuner for MB-sPCA
#'
#' @description
#' `TunerSeqMBsPCA` tunes one component at a time for `PipeOpMBsPCA`, choosing
#' block-wise sparsity budgets by inner resampling and deflation. Candidate
#' solutions are now scored with the package's MB-sPCA measure
#' `mbspca.mean_ev`, i.e. the same prediction-side explained-variance quantity
#' exposed to users, rather than a separate hard-coded proxy objective.
#'
#' The tuner supports only the package-native MB-sPCA measure
#' `mbspca.mean_ev`. Passing any other measure now errors explicitly.
#'
#' @section Works with:
#' A learner whose pipeline contains a `PipeOpMBsPCA` node.
#'
#' @section Construction:
#' `TunerSeqMBsPCA$new(tuner = "random_search", budget = 100L,`
#' `resampling = mlr3::rsmp("cv", folds = 3), parallel = "none",`
#' `early_stopping = TRUE, n_perm = 1000L, perm_alpha = 0.05)`
#'
#' @param tuner (`character(1)`) Optimizer ID for the inner single-component
#'   search.
#' @param budget (`integer(1)`) Number of evaluations for each component-wise
#'   search.
#' @param resampling (`Resampling`) Template for inner CV.
#' @param parallel (`character(1)`) `"none"` or `"inner"`.
#' @param early_stopping (`logical(1)`) Perform a permutation test after each
#'   component and stop if not significant (PC-1 is always kept).
#' @param n_perm (`integer(1)`) Number of permutations for the test.
#' @param perm_alpha (`numeric(1)`) Alpha-level for the test.
#'
#' @return
#' The tuned `TuningInstance` invisibly. The result is written via
#' `assign_result()` with `learner_param_vals = list(c_matrix = <matrix>)` and
#' the aggregated value of `mbspca.mean_ev` for the selected `c_matrix`.
#'
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3pipelines)
#' library(mlr3tuning)
#' blocks = list(eng = c("disp", "hp", "drat"), body = c("wt", "qsec"))
#' po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 3))
#' lrn = as_learner(po %>>% po("learner", lrn("clust.kmeans", centers = 2)))
#'
#' ti = TuningInstanceSingleCrit$new(
#'   task = mlr3cluster::TaskClust$new("x", backend = mtcars),
#'   learner = lrn,
#'   resampling = rsmp("holdout"),
#'   measure = msr("mbspca.mean_ev"),
#'   search_space = paradox::ps(),
#'   terminator = trm("none")
#' )
#' TunerSeqMBsPCA$new(budget = 50)$optimize(ti)
#' ti$result_learner_param_vals$c_matrix
#' }
#'
#' @seealso [PipeOpMBsPCA], [mlr3tuning::Tuner], [bbotk]
#' @family mb-sPCA
#' @importFrom mlr3 rsmp
#' @import lgr
#' @export
TunerSeqMBsPCA = R6::R6Class(
  "TunerSeqMBsPCA",
  inherit = mlr3tuning::Tuner,

  public = list(
    #' @description Create a new TunerSeqMBsPCA.
    initialize = function(tuner = "random_search",
      budget = 100L,
      resampling = rsmp("cv", folds = 3),
      parallel = "none",
      early_stopping = TRUE,
      n_perm = 1000L,
      perm_alpha = 0.05) {

      checkmate::assert_choice(parallel, c("none", "inner"))
      checkmate::assert_int(budget, lower = 1L)
      checkmate::assert_flag(early_stopping)
      checkmate::assert_int(n_perm, lower = 1L)
      checkmate::assert_number(perm_alpha, lower = 0, upper = 1)

      if (grepl("async", tuner, ignore.case = TRUE)) {
        lgr$warning("Asynchronous tuners are unsupported - using 'random_search' instead.")
        tuner = "random_search"
      }

      private$.tuner = tuner
      private$.budget = budget
      private$.resampling_tpl = resampling
      private$.parallel = parallel
      private$.early_stop = early_stopping
      private$.n_perm = n_perm
      private$.perm_alpha = perm_alpha

      super$initialize(
        param_set = paradox::ps(),
        properties = "single-crit",
        param_classes = c("ParamInt", "ParamDbl", "ParamLgl", "ParamFct")
      )
    },

    #' @description Run the sequential tuning loop.
    #' @param instance [mlr3tuning::TuningInstanceSingleCrit] (or compatible).
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

    .resolve_measure = function(inst) {
      measure = tryCatch(inst$objective$measure, error = function(e) NULL)
      if (is.null(measure)) {
        measure = tryCatch(inst$objective$measures[[1L]], error = function(e) NULL)
      }
      if (is.null(measure)) {
        stop("TunerSeqMBsPCA requires the measure 'mbspca.mean_ev'.", call. = FALSE)
      }
      key = .mbspca_measure_key(measure)
      if (!identical(key, "mbspca.mean_ev")) {
        stop("TunerSeqMBsPCA supports only the measure 'mbspca.mean_ev'.", call. = FALSE)
      }
      measure
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
        stop("TunerSeqMBsPCA does not support custom measure aggregators.", call. = FALSE)
      }
      aggr = measure$aggregator %||% function(x) mean(x)
      aggr(scores)
    },

    .empty_fold_payload = function(block_names) {
      list(
        ev_comp = numeric(),
        ev_block = matrix(numeric(), nrow = 0L, ncol = length(block_names),
          dimnames = list(NULL, block_names)),
        blocks = block_names
      )
    },

    .append_fold_payload = function(payload, payload_k) {
      payload$ev_comp = c(as.numeric(payload$ev_comp), as.numeric(payload_k$ev_comp))
      payload$ev_block = rbind(payload$ev_block, as.matrix(payload_k$ev_block))
      payload
    },

    .trim_last_component = function(payload) {
      if (length(payload$ev_comp)) {
        payload$ev_comp = payload$ev_comp[-length(payload$ev_comp)]
      }
      if (!is.null(payload$ev_block) && nrow(payload$ev_block) > 0L) {
        payload$ev_block = payload$ev_block[seq_len(nrow(payload$ev_block) - 1L), , drop = FALSE]
      }
      payload
    },

    .pre_graph_before_mbspca = function(learner, mbspca_id = NULL) {
      ids = learner$graph$ids()
      mbspca_id = mbspca_id %||% .find_pipeop_id_by_class(
        learner$graph,
        class_name = "PipeOpMBsPCA",
        where = "learner$graph"
      )
      pos = match(mbspca_id, ids)
      if (is.na(pos) || pos == 1L) {
        return(mlr3pipelines::Graph$new())
      }
      mlr3pipelines::as_graph(
        learner$graph$clone(deep = TRUE)$pipeops[ids[seq_len(pos - 1L)]]
      )
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
          stop(sprintf("After preprocessing, no columns matched any of: %s", paste(cols, collapse = ", ")), call. = FALSE)
        }
        M = as.matrix(data[, ..ex])
        storage.mode(M) = "double"
        M
      })
      names(out) = names(block_map)
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

    .deflate_blocks_val = function(X_tr, X_val, W_list) {
      P_fit = private$.compute_train_loadings(X_tr, W_list)
      for (b in seq_along(X_val)) {
        tb = X_val[[b]] %*% W_list[[b]]
        X_val[[b]] = X_val[[b]] - tcrossprod(tb, P_fit[[b]])
      }
      X_val
    },

    .one_lv_payload = function(X_train_fit, X_test, W_list) {
      P_fit = private$.compute_train_loadings(X_train_fit, W_list)
      res = compute_test_ev(
        X_blocks_test = X_test,
        W_all = list(W_list),
        P_all = list(P_fit),
        deflate = TRUE,
        performance_metric = "mac",
        correlation_method = "pearson",
        loading_source = "train",
        clamp_ev = "none"
      )
      list(
        ev_comp = as.numeric(res$ev_comp),
        ev_block = as.matrix(res$ev_block),
        blocks = names(X_test)
      )
    },

    .run = function(inst) {

      measure = private$.resolve_measure(inst)
      learner_tpl = inst$objective$learner
      task_full = inst$objective$task$clone(deep = TRUE)

      mbspca_id = .find_pipeop_id_by_class(
        learner_tpl$graph,
        class_name = "PipeOpMBsPCA",
        where = "learner_tpl$graph"
      )
      pre_graph_tpl = private$.pre_graph_before_mbspca(learner_tpl, mbspca_id = mbspca_id)
      blocks_raw = learner_tpl$graph$pipeops[[mbspca_id]]$blocks
      K_max = learner_tpl$graph$pipeops[[mbspca_id]]$param_set$values$ncomp %||% 1L
      B = length(blocks_raw)
      if (B == 0L) stop("PipeOpMBsPCA has no blocks defined.", call. = FALSE)

      if (length(pre_graph_tpl$pipeops)) {
        pre_graph_full = pre_graph_tpl$clone(deep = TRUE)
        pre_graph_full$train(task_full)
        df_full = data.table::last(pre_graph_full$predict(task_full))$data()
      } else {
        df_full = task_full$data()
      }
      blocks = lapply(blocks_raw, function(cols) mb_expand_block_cols(names(df_full), cols))
      names(blocks) = names(blocks_raw)
      X_residual = private$.make_blocks(df_full, blocks, allow_encoded = FALSE)
      names(X_residual) = names(blocks)

      rs = private$.resampling_tpl$clone()
      if (!rs$is_instantiated) rs$instantiate(task_full)

      fold_tr = vector("list", rs$iters)
      fold_val = vector("list", rs$iters)
      for (f in seq_len(rs$iters)) {
        task_tr = task_full$clone(deep = FALSE)$filter(rs$train_set(f))
        task_va = task_full$clone(deep = FALSE)$filter(rs$test_set(f))

        g = pre_graph_tpl$clone(deep = TRUE)
        if (length(g$pipeops)) {
          df_tr = g$train(task_tr)[[1L]]$data()
          df_va = g$predict(task_va)[[1L]]$data()
        } else {
          df_tr = task_tr$data()
          df_va = task_va$data()
        }

        fold_tr[[f]] = private$.make_blocks(df_tr, blocks, allow_encoded = FALSE)
        fold_val[[f]] = private$.make_blocks(df_va, blocks, allow_encoded = FALSE)
      }

      if (private$.parallel == "inner") {
        if (!requireNamespace("future", quietly = TRUE) || !requireNamespace("future.apply", quietly = TRUE)) {
          stop(
            "parallel = 'inner' requires the optional packages 'future' and 'future.apply'.",
            call. = FALSE
          )
        }
        future::plan("multisession", workers = max(1L, future::availableCores() - 1L))
        on.exit(future::plan("sequential"), add = TRUE)
        fold_apply = function(X, FUN) future.apply::future_sapply(X, FUN, future.seed = TRUE)
      } else {
        fold_apply = function(X, FUN) sapply(X, FUN)
      }

      C_star = matrix(NA_real_, B, K_max,
        dimnames = list(names(blocks), paste0("PC", seq_len(K_max))))
      fold_payloads = lapply(seq_len(rs$iters), function(i) private$.empty_fold_payload(names(blocks)))
      n_val = vapply(seq_len(rs$iters), function(f) nrow(fold_val[[f]][[1L]]), integer(1L))

      for (k in seq_len(K_max)) {
        lgr$info("-> MB-sPCA component %d / %d", k, K_max)

        ps_k = do.call(
          paradox::ps,
          setNames(lapply(names(blocks), function(bn) {
            paradox::p_int(lower = 1L, upper = ncol(X_residual[[bn]]))
          }), paste0("c_", names(blocks)))
        )

        cache = new.env(parent = emptyenv())
        obj_fun = bbotk::ObjectiveRFun$new(
          fun = function(xs) {
            key = paste(unlist(xs, use.names = FALSE), collapse = "_")
            if (exists(key, envir = cache, inherits = FALSE)) {
              return(list(Score = cache[[key]]))
            }

            c_vec = sqrt(unlist(xs, use.names = FALSE))
            fold_scores = fold_apply(seq_len(rs$iters), function(f) {
              fit = cpp_mbspca_one_lv(fold_tr[[f]], c_vec, max_iter = 50L, tol = 1e-4)
              payload = private$.one_lv_payload(fold_tr[[f]], fold_val[[f]], fit$W)
              mbspca_measure_score_from_payload(payload, measure)
            })
            score_raw = private$.aggregate_scores(fold_scores, measure, n_obs = n_val)
            score_opt = if (isTRUE(measure$minimize)) -score_raw else score_raw
            cache[[key]] = score_opt
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
        lgr$info("   chosen c-vector: %s", paste(C_star[, k], collapse = ", "))

        for (f in seq_len(rs$iters)) {
          Xtr_before = fold_tr[[f]]
          Xva_before = fold_val[[f]]
          fit_fold_k = cpp_mbspca_one_lv(Xtr_before, C_star[, k], max_iter = 50L, tol = 1e-4)
          payload_k = private$.one_lv_payload(Xtr_before, Xva_before, fit_fold_k$W)
          fold_payloads[[f]] = private$.append_fold_payload(fold_payloads[[f]], payload_k)
          fold_val[[f]] = private$.deflate_blocks_val(Xtr_before, Xva_before, fit_fold_k$W)
          fold_tr[[f]] = private$.deflate_blocks(Xtr_before, fit_fold_k$W)
        }

        fit_full = cpp_mbspca_one_lv(X_residual, C_star[, k], max_iter = 50L, tol = 1e-4)

        if (private$.early_stop) {
          p_val = perm_test_component_mbspca(
            X_residual, fit_full$W, C_star[, k],
            n_perm = private$.n_perm, alpha = private$.perm_alpha
          )
          lgr$info("   permutation p-value = %.4g", p_val)

          if (p_val > private$.perm_alpha) {
            lgr$info("   early stop triggered (component not significant)")
            if (k > 1L) {
              C_star = C_star[, seq_len(k - 1L), drop = FALSE]
              fold_payloads = lapply(fold_payloads, private$.trim_last_component)
            } else {
              C_star = C_star[, 1L, drop = FALSE]
            }
            break
          }
        }

        X_residual = private$.deflate_blocks(X_residual, fit_full$W)
      }

      fold_scores_final = vapply(fold_payloads, function(pl) {
        mbspca_measure_score_from_payload(pl, measure)
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
