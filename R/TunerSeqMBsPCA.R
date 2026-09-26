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
#' @param early_stopping (`logical(1)`) Perform a conditional permutation
#'   diagnostic after each component and stop if its cutoff is not met (PC-1 is
#'   always kept). This is not full-pipeline confirmatory inference.
#' @param n_perm (`integer(1)`) Number of permutations for the diagnostic.
#' @param perm_alpha (`numeric(1)`) Cutoff for the diagnostic.
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
#' po_mbspca = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2L))
#' learner = as_learner(
#'   po_mbspca %>>% po("learner", lrn("clust.kmeans", centers = 2L))
#' )
#'
#' instance = ti(
#'   task = mlr3cluster::TaskClust$new("x", backend = mtcars),
#'   learner = learner,
#'   resampling = rsmp("insample"),
#'   measure = msr("mbspca.mean_ev"),
#'   terminator = trm("evals", n_evals = 1L)
#' )
#' TunerSeqMBsPCA$new(
#'   budget = 2L,
#'   resampling = rsmp("cv", folds = 2L),
#'   early_stopping = FALSE
#' )$optimize(instance)
#' instance$result$learner_param_vals[[1L]]$c_matrix
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
        stop(
          "Asynchronous tuners are unsupported by TunerSeqMBsPCA because the sequential component path depends on deterministic synchronous evaluations. Choose a synchronous tuner such as 'random_search'.",
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
      mbspca_id = mbspca_id %||% .find_pipeop_id_by_class(
        learner$graph,
        class_name = "PipeOpMBsPCA",
        where = "learner$graph"
      )
      mb_preprocessing_graph(learner$graph, mbspca_id)
    },

    .resolve_blocks = function(data, block_map) {
      resolved = lapply(block_map, function(cols) {
        candidates = mb_expand_block_cols(names(data), cols)
        candidates[vapply(candidates, function(column) {
          is.numeric(data[[column]]) && mb_has_finite_variance(data[[column]])
        }, logical(1L))]
      })
      empty = names(resolved)[!lengths(resolved)]
      if (length(empty)) {
        stop(sprintf(
          "No usable training features remain in blocks: %s.",
          paste(empty, collapse = ", ")
        ), call. = FALSE)
      }
      resolved
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
        if (!length(ex)) {
          stop(sprintf("After preprocessing, no columns matched any of: %s", paste(cols, collapse = ", ")), call. = FALSE)
        }
        mb_assert_columns_present(
          colnames_dt = names(data),
          required = ex,
          context = "Preprocessed data for TunerSeqMBsPCA",
          hint = "Ensure that all block columns survive upstream preprocessing exactly as during tuning."
        )
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

    .one_lv_payload = function(X_train_fit, X_test, W_list, original_ss = NULL) {
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
      component_po = learner_tpl$graph$pipeops[[mbspca_id]]
      po_vals = utils::modifyList(paradox::default_values(component_po$param_set),
        component_po$param_set$values, keep.null = TRUE)
      blocks_raw = component_po$blocks
      K_max = po_vals$ncomp
      max_iter = po_vals$max_iter
      tol = po_vals$tol
      B = length(blocks_raw)
      if (B == 0L) stop("PipeOpMBsPCA has no blocks defined.", call. = FALSE)

      if (length(pre_graph_tpl$pipeops)) {
        pre_graph_full = pre_graph_tpl$clone(deep = TRUE)
        df_full = pre_graph_full$train(task_full)[[1L]]$data()
      } else {
        df_full = task_full$data()
      }
      blocks = private$.resolve_blocks(df_full, blocks_raw)
      names(blocks) = names(blocks_raw)
      X_residual = private$.make_blocks(df_full, blocks, allow_encoded = FALSE)
      names(X_residual) = names(blocks)

      rs = private$.resampling_tpl$clone()
      if (!rs$is_instantiated) rs$instantiate(task_full)

      fold_tr = vector("list", rs$iters)
      fold_val = vector("list", rs$iters)
      for (f in seq_len(rs$iters)) {
        mb_assert_resampling_split(task_full, rs$train_set(f), rs$test_set(f))
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

        fold_blocks = private$.resolve_blocks(df_tr, blocks_raw)
        fold_tr[[f]] = private$.make_blocks(df_tr, fold_blocks, allow_encoded = FALSE)
        fold_val[[f]] = private$.make_blocks(df_va, fold_blocks, allow_encoded = FALSE)
      }

      fold_ss = lapply(fold_val, function(X) {
        vapply(X, function(block) sum(block^2), numeric(1L))
      })
      budget_width = vapply(names(blocks), function(block) {
        min(c(ncol(X_residual[[block]]), vapply(fold_tr, function(X) {
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
            paradox::p_int(lower = 1L, upper = budget_width[[bn]])
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
              fit = cpp_mbspca_one_lv(fold_tr[[f]], c_vec, max_iter = max_iter, tol = tol)
              payload = private$.one_lv_payload(fold_tr[[f]], fold_val[[f]], fit$W, original_ss = fold_ss[[f]])
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
          fit_fold_k = cpp_mbspca_one_lv(Xtr_before, C_star[, k], max_iter = max_iter, tol = tol)
          payload_k = private$.one_lv_payload(Xtr_before, Xva_before, fit_fold_k$W, original_ss = fold_ss[[f]])
          fold_payloads[[f]] = private$.append_fold_payload(fold_payloads[[f]], payload_k)
          fold_val[[f]] = private$.deflate_blocks_val(Xtr_before, Xva_before, fit_fold_k$W)
          fold_tr[[f]] = private$.deflate_blocks(Xtr_before, fit_fold_k$W)
        }

        fit_full = cpp_mbspca_one_lv(X_residual, C_star[, k], max_iter = max_iter, tol = tol)

        if (private$.early_stop) {
          p_val = perm_test_component_mbspca(
            X_residual, fit_full$W, C_star[, k],
            n_perm = private$.n_perm, alpha = private$.perm_alpha,
            max_iter = max_iter, tol = tol
          )
          lgr$info("   permutation p-value = %.4g", p_val)

          if (p_val > private$.perm_alpha) {
            lgr$info("   early stop triggered (conditional diagnostic cutoff not met)")
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
