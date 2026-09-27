#' @title Sequential Component-wise Tuner for MB-sPCA
#'
#' @description
#' `TunerSeqMBsPCA` tunes one component at a time for `PipeOpMBsPCA`, choosing
#' block-wise sparsity budgets by inner resampling and deflation. Candidate
#' solutions are scored with the package's MB-sPCA measure `mbspca.mean_ev`,
#' i.e. the same prediction-side explained-variance quantity exposed to users,
#' rather than a separate hard-coded proxy objective.
#'
#' The tuner supports only the package-native MB-sPCA measure
#' `mbspca.mean_ev`. Passing any other measure errors explicitly.
#'
#' @details
#' **Inner folds mirror the final fit.** For every inner fold, the
#' preprocessing graph upstream of `PipeOpMBsPCA` is trained on the fold's
#' training rows only and applied to its validation rows. Block columns are
#' resolved exactly as `PipeOpMBsPCA` resolves them, from the feature columns it
#' receives (never the target, and only those selected by its
#' `affect_columns`): [mb_resolve_block_columns()] (so encoded factor columns
#' belong to their declared block), followed by the numeric and
#' finite-variance filters. Blocks without a usable column on the full tuning
#' task are dropped with a warning, as `PipeOpMBsPCA` drops them, and the tuned
#' `c_matrix` has one row per retained block. As in `PipeOpMBsPCA`, the node's
#' `ncomp` may not exceed the rank of any centred retained block on the full
#' tuning task; this is checked before the search. Every retained block must
#' keep at least one usable column in each inner training fold. Each
#' fold's training blocks are centred by their column means and its validation
#' rows by the same means, as `PipeOpMBsPCA` centres prediction data by its
#' training means. Fold fits use the `max_iter` and `tol` of the `PipeOpMBsPCA`
#' node. The solver starts deterministically, so the fits that scored the
#' selected candidate are reused for the fold payloads and the deflation. A
#' warning names retained components whose fold fits did not converge. A
#' candidate whose score is undefined in some fold ranks last; if every
#' candidate of a component is undefined, the tuning stops with an error. An
#' error raised while scoring a candidate stops the tuning with its original
#' message.
#'
#' `result_y` is the inner-resampling score of the tuned `c_matrix`; since each
#' component's `c` maximises this score on the same folds, it is
#' optimistically biased.
#'
#' **Early stopping is a heuristic.** After component `k` is selected, it is
#' refitted on the full tuning task (centred by its own column means and
#' deflated by the earlier full-data components) and tested with
#' `perm_test_component_mbspca()`. That test permutes the rows of every block
#' independently, which keeps each block's own covariance and destroys only the
#' association between blocks. Its null is therefore "no cross-block
#' association": a component that captures strong block-specific variance but
#' little shared structure does not pass and stops the tuning. The test needs
#' at least two blocks; with a single retained block early stopping is skipped
#' with a warning and all `ncomp` components are tuned. The test reuses the data
#' that selected `c`, so `perm_alpha` is a cutoff, not an error rate.
#'
#' After `optimize()`, the field `diagnostics` holds the resolved blocks, one
#' row per component and inner fold (`fold_statistics`) and one row per
#' evaluated component (`components`).
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
#' @param parallel (`character(1)`) `"none"` or `"inner"`, which fits the
#'   inner folds of each candidate in parallel with `future` (multisession).
#'   Fold fits draw no random numbers, so both settings propose the same
#'   candidates for a given seed.
#' @param early_stopping (`logical(1)`) Run the cross-block permutation
#'   diagnostic after each component and stop if its cutoff is not met (PC-1 is
#'   always kept; see Details). It requires at least two blocks and is skipped
#'   with a warning otherwise. This is a tuning heuristic, not confirmatory
#'   inference.
#' @param n_perm (`integer(1)`) Number of permutations for the diagnostic.
#' @param perm_alpha (`numeric(1)`) Cutoff for the diagnostic; not an error
#'   rate.
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

  active = list(
    #' @field diagnostics (`list()` or `NULL`)\cr
    #' Diagnostics of the last completed `optimize()` call: `blocks`, the
    #' resolved block columns on the full tuning task; `fold_statistics`, a
    #' [data.table::data.table()] with one row per evaluated component and inner
    #' fold (`component`, `fold`, `n_train`, `n_validation`, the fold `score`
    #' of the selected `c`, solver `converged`, then `iterations` and
    #' `p_value`, which are `NA` for MB-sPCA folds, and `kept`); and
    #' `components`, one row per evaluated component with its `c_<block>`
    #' values, `search_score`, `n_evals`, the early-stopping `p_value` (`NA`
    #' when the diagnostic was not run) and `kept`. Read-only.
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
    .diagnostics = NULL,

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
      private$.diagnostics = NULL
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
      # Same effective value the PipeOp fits (a param_set override wins).
      blocks_raw = po_vals$blocks %||% component_po$blocks
      K_max = po_vals$ncomp
      max_iter = po_vals$max_iter
      tol = po_vals$tol
      if (!length(blocks_raw)) stop("PipeOpMBsPCA has no blocks defined.", call. = FALSE)

      if (length(pre_graph_tpl$pipeops)) {
        pre_graph_full = pre_graph_tpl$clone(deep = TRUE)
        df_full = .mb_tuner_feature_data(pre_graph_full$train(task_full)[[1L]], component_po)
      } else {
        df_full = .mb_tuner_feature_data(task_full, component_po)
      }
      blocks = .mb_tuner_retained_blocks(df_full, blocks_raw, "PipeOpMBsPCA")
      B = length(blocks)
      # Full-task blocks, centred by their own means as PipeOpMBsPCA centres its
      # training data. They pass the PipeOp's rank guard and feed the
      # early-stopping diagnostic.
      X_full = .mb_tuner_assert_rank(df_full, blocks, K_max, "PipeOpMBsPCA")

      early_stop = private$.early_stop
      if (early_stop && B < 2L) {
        warning(
          "early_stopping requires at least two blocks: the MB-sPCA permutation diagnostic tests cross-block association, which a single block cannot carry. Early stopping is skipped and all requested components are tuned.",
          call. = FALSE
        )
        early_stop = FALSE
      }

      X_residual = if (early_stop) X_full

      rs = private$.resampling_tpl$clone()
      if (!rs$is_instantiated) rs$instantiate(task_full)
      n_folds = rs$iters

      fold_tr = vector("list", n_folds)
      fold_val = vector("list", n_folds)
      for (f in seq_len(n_folds)) {
        mb_assert_resampling_split(task_full, rs$train_set(f), rs$test_set(f))
        task_tr = task_full$clone(deep = FALSE)$filter(rs$train_set(f))
        task_va = task_full$clone(deep = FALSE)$filter(rs$test_set(f))

        g = pre_graph_tpl$clone(deep = TRUE)
        if (length(g$pipeops)) {
          df_tr = .mb_tuner_feature_data(g$train(task_tr)[[1L]], component_po)
          df_va = .mb_tuner_feature_data(g$predict(task_va)[[1L]], component_po)
        } else {
          df_tr = .mb_tuner_feature_data(task_tr, component_po)
          df_va = .mb_tuner_feature_data(task_va, component_po)
        }

        fold_blocks = .mb_tuner_fold_blocks(df_tr, blocks_raw, names(blocks), f)
        centred = .mb_center_fold_blocks(
          .mb_tuner_block_matrices(df_tr, fold_blocks, sprintf("inner training fold %d", f)),
          .mb_tuner_block_matrices(df_va, fold_blocks, sprintf("inner validation fold %d", f))
        )
        fold_tr[[f]] = centred$train
        fold_val[[f]] = centred$validation
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

      # One-component fit of the current (deflated) fold data, its held-out
      # payload and score.
      fit_fold = function(f, c_vec) {
        fit = cpp_mbspca_one_lv(fold_tr[[f]], c_vec, max_iter = max_iter, tol = tol)
        payload = private$.one_lv_payload(fold_tr[[f]], fold_val[[f]], fit$W, original_ss = fold_ss[[f]])
        list(fit = fit, payload = payload, score = mbspca_measure_score_from_payload(payload, measure))
      }

      C_star = matrix(NA_real_, B, K_max,
        dimnames = list(names(blocks), paste0("PC", seq_len(K_max))))
      fold_payloads = lapply(seq_len(n_folds), function(i) private$.empty_fold_payload(names(blocks)))
      n_val = vapply(seq_len(n_folds), function(f) nrow(fold_val[[f]][[1L]]), integer(1L))
      n_fit = vapply(seq_len(n_folds), function(f) nrow(fold_tr[[f]][[1L]]), integer(1L))
      fold_stats = list()
      component_stats = list()

      for (k in seq_len(K_max)) {
        lgr$info("-> MB-sPCA component %d / %d", k, K_max)

        ps_k = do.call(
          paradox::ps,
          setNames(lapply(names(blocks), function(bn) {
            paradox::p_int(lower = 1L, upper = budget_width[[bn]])
          }), paste0("c_", names(blocks)))
        )

        cache = new.env(parent = emptyenv())
        # Fold fits of the best candidate so far, reused after the search, and
        # the first error raised while scoring a candidate.
        best = new.env(parent = emptyenv())
        best$key = NULL
        best$score = -Inf
        best$error = NULL
        obj_fun = bbotk::ObjectiveRFun$new(
          fun = function(xs) {
            key = paste(unlist(xs, use.names = FALSE), collapse = "_")
            if (exists(key, envir = cache, inherits = FALSE)) {
              return(list(Score = cache[[key]]))
            }

            c_vec = sqrt(unlist(xs, use.names = FALSE))
            fold_results = tryCatch(
              fold_map(seq_len(n_folds), function(f) fit_fold(f, c_vec)),
              error = function(e) {
                best$error = e
                stop(e)
              }
            )
            fold_scores = vapply(fold_results, function(res) as.numeric(res$score), numeric(1L))
            score_raw = private$.aggregate_scores(fold_scores, measure, n_obs = n_val)
            score_opt = if (isTRUE(measure$minimize)) -score_raw else score_raw
            # An undefined score (e.g. no explained variance is defined in a
            # fold) ranks the candidate last instead of aborting the search.
            if (!is.finite(score_opt)) {
              score_opt = -Inf
            }
            cache[[key]] = score_opt
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
        if (!is.null(best$error)) {
          .mb_tuner_rethrow(best$error, sprintf("TunerSeqMBsPCA failed while scoring component %d", k))
        }
        cached_scores = unlist(mget(ls(cache, all.names = TRUE), envir = cache), use.names = FALSE)
        if (length(cached_scores) && all(cached_scores %in% -Inf)) {
          stop(sprintf(
            "All %d candidate c-vectors produced an undefined '%s' score for component %d. Increase data support in the inner folds.",
            length(cached_scores),
            measure$id,
            k
          ), call. = FALSE)
        }
        if (!is.null(optimize_error)) {
          .mb_tuner_rethrow(optimize_error, sprintf("TunerSeqMBsPCA failed while tuning component %d", k))
        }

        x_star = unlist(inst_k$result_x_domain, use.names = FALSE)
        C_star[, k] = sqrt(x_star)
        lgr$info("   chosen c-vector: %s", paste(C_star[, k], collapse = ", "))
        search_score = as.numeric(inst_k$result_y)
        if (isTRUE(measure$minimize)) {
          search_score = -search_score
        }

        # The deterministic solver reproduces the search fits of the selected
        # candidate, so they are reused rather than refitted.
        folds_k = if (identical(best$key, paste(x_star, collapse = "_"))) {
          best$folds
        } else {
          lapply(seq_len(n_folds), function(f) fit_fold(f, C_star[, k]))
        }

        for (f in seq_len(n_folds)) {
          X_tr_before = fold_tr[[f]]
          W_k = folds_k[[f]]$fit$W
          fold_payloads[[f]] = private$.append_fold_payload(fold_payloads[[f]], folds_k[[f]]$payload)
          fold_val[[f]] = private$.deflate_blocks_val(X_tr_before, fold_val[[f]], W_k)
          fold_tr[[f]] = private$.deflate_blocks(X_tr_before, W_k)
        }

        fold_stats[[k]] = data.table::data.table(
          component = k,
          fold = seq_len(n_folds),
          n_train = n_fit,
          n_validation = n_val,
          score = vapply(folds_k, function(res) as.numeric(res$score), numeric(1L)),
          converged = vapply(folds_k, function(res) as.logical(res$fit$converged %||% NA)[1L], logical(1L)),
          iterations = NA_integer_,
          p_value = NA_real_
        )

        p_val = NA_real_
        stop_here = FALSE
        if (early_stop) {
          fit_full = cpp_mbspca_one_lv(X_residual, C_star[, k], max_iter = max_iter, tol = tol)
          p_val = perm_test_component_mbspca(
            X_residual, fit_full$W, C_star[, k],
            n_perm = private$.n_perm, alpha = private$.perm_alpha,
            max_iter = max_iter, tol = tol
          )
          lgr$info("   permutation p-value = %.4g", p_val)
          stop_here = p_val > private$.perm_alpha
          X_residual = private$.deflate_blocks(X_residual, fit_full$W)
        }
        component_stats[[k]] = cbind(
          data.table::data.table(component = k),
          data.table::as.data.table(as.list(stats::setNames(C_star[, k], paste0("c_", names(blocks))))),
          data.table::data.table(
            search_score = search_score,
            n_evals = nrow(inst_k$archive$data),
            p_value = p_val
          )
        )

        if (stop_here) {
          lgr$info("   early stop triggered (diagnostic cutoff not met)")
          if (k > 1L) {
            C_star = C_star[, seq_len(k - 1L), drop = FALSE]
            fold_payloads = lapply(fold_payloads, private$.trim_last_component)
          } else {
            C_star = C_star[, 1L, drop = FALSE]
          }
          break
        }
      }

      n_kept = ncol(C_star)
      fold_statistics = data.table::rbindlist(fold_stats)
      fold_statistics$kept = fold_statistics$component <= n_kept
      components = data.table::rbindlist(component_stats)
      components$kept = components$component <= n_kept
      .mb_tuner_warn_nonconverged(
        fold_statistics,
        sprintf("TunerSeqMBsPCA: the MB-sPCA solver did not converge within %d iterations", max_iter),
        "PC"
      )

      fold_scores_final = vapply(fold_payloads, function(pl) {
        mbspca_measure_score_from_payload(pl, measure)
      }, numeric(1L))
      y_raw = private$.aggregate_scores(fold_scores_final, measure, n_obs = n_val)
      if (!is.finite(y_raw)) {
        stop(sprintf(
          "The selected c-matrix produced an undefined '%s' score in the inner validation folds.",
          measure$id
        ), call. = FALSE)
      }

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
