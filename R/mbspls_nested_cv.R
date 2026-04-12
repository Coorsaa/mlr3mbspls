# Internal helpers shared by the direct and batchtools nested-CV paths.
mbspls_nested_cv_result_row = function(split_id, inner_score, payload, performance_metric, c_star = NULL) {
  if (is.null(payload)) {
    return(data.table::data.table(
      split = split_id,
      inner_score = inner_score,
      mac_lv1_test = NA_real_,
      mac_evwt_test = NA_real_,
      mac_evwt_defined = FALSE,
      mac_evwt_status = "Prediction payload is missing.",
      ncomp_kept = if (is.null(c_star)) NA_integer_ else ncol(c_star),
      perf_metric = performance_metric,
      val_p_last = NA_real_
    ))
  }

  mac = as.numeric(payload$mac_comp %||% numeric())
  vp = payload$val_test_p %||% payload$val_perm_p
  vs = payload$val_test_stat %||% NULL
  score_info = mbspls_measure_score_diagnostics(payload, "mbspls.mac_evwt")

  data.table::data.table(
    split = split_id,
    inner_score = inner_score,
    mac_lv1_test = if (length(mac)) mac[1L] else NA_real_,
    mac_evwt_test = score_info$score,
    mac_evwt_defined = isTRUE(score_info$defined),
    mac_evwt_status = if (isTRUE(score_info$defined)) NA_character_ else as.character(score_info$message %||% "Measure 'mbspls.mac_evwt' returned a non-finite score."),
    ncomp_kept = if (length(mac)) length(mac) else if (is.null(c_star)) NA_integer_ else ncol(c_star),
    perf_metric = payload$perf_metric %||% performance_metric,
    val_p_last = if (!is.null(vp)) as.numeric(utils::tail(vp, 1L)) else NA_real_,
    val_p_all = list(vp),
    val_stat_all = list(vs)
  )
}

mbspls_metric_summary = function(x) {
  x = as.numeric(x)
  n_total = length(x)
  finite = x[is.finite(x)]
  n_defined = length(finite)
  n_failed = n_total - n_defined

  if (!n_defined) {
    return(data.table::data.table(
      mean = NA_real_,
      sd = NA_real_,
      median = NA_real_,
      q05 = NA_real_,
      q95 = NA_real_,
      min = NA_real_,
      max = NA_real_,
      n = 0L,
      n_total = n_total,
      n_defined = n_defined,
      n_failed = n_failed
    ))
  }

  data.table::data.table(
    mean = mean(finite),
    sd = stats::sd(finite),
    median = stats::median(finite),
    q05 = as.numeric(stats::quantile(finite, 0.05)),
    q95 = as.numeric(stats::quantile(finite, 0.95)),
    min = min(finite),
    max = max(finite),
    n = n_defined,
    n_total = n_total,
    n_defined = n_defined,
    n_failed = n_failed
  )
}

mbspls_metric_summary_row = function(label, x) {
  cbind(
    data.table::data.table(metric = label),
    mbspls_metric_summary(x)
  )
}

#' @title Nested CV for unsupervised MB-sPLS
#'
#' @description
#' Perform nested cross-validation for unsupervised multi-block sparse PLS
#' (MB-sPLS) using a specified resampling strategy.
#' This function tunes the model on inner folds and evaluates it on outer folds,
#' returning a summary of results.
#'
#' @param task [mlr3::Task] with all pre-MBsPLS features.
#' @param graphlearner [mlr3pipelines::GraphLearner] with a PipeOpMBsPLS node.
#' @param rs_outer [mlr3::Resampling] instantiated (outer) resampling.
#' @param rs_inner [mlr3::Resampling] template (inner) resampling.
#' @param ncomp integer, max #components considered by the tuner.
#' @param tuner_budget integer, #candidate evaluations per component.
#' @param tuning_early_stop logical, stop tuning early if no sig. components.
#' @param performance_metric "mac" or "frobenius" for the latent correlation.
#' @param val_test "none", "permutation", or "bootstrap" - run on OUTER test.
#' @param val_test_n integer, permutations/boot reps on OUTER test.
#' @param val_test_alpha numeric, early-stop / CI level param for validation tests.
#' @param val_permute_all logical, permute all blocks in test validation.
#' @param n_perm_tuning integer, permutations used inside the tuner's early stop.
#' @param perm_alpha_tuning numeric, tuner early-stop alpha (component-wise).
#' @param store_payload logical, keep the full PipeOpMBsPLS predict payload.
#'
#' @return list with:
#'   - results: data.table with per-outer-split summary (inner_score, test MACs,
#'     and EV-weighted-MAC status columns `mac_evwt_defined` / `mac_evwt_status`)
#'   - c_mats: list of tuned C* matrices per outer split
#'   - inner_scores: numeric vector of best inner-CV scores (one per outer split)
#'   - payloads: (optional) list of PipeOp logging payloads for each outer test
#'   - summary_table: data.table with aggregated metrics across outer splits,
#'     including `n_total`, `n_defined`, and `n_failed`
#'
#' @importFrom lgr lgr
#' @export
mbspls_nested_cv = function(
  task,
  graphlearner,
  rs_outer,
  rs_inner,
  ncomp,
  tuner_budget,
  tuning_early_stop = TRUE,
  performance_metric = c("mac", "frobenius"),
  val_test = c("none", "permutation", "bootstrap"),
  val_test_n = 1000L,
  val_test_alpha = 0.05,
  val_permute_all = TRUE,
  n_perm_tuning = 500L,
  perm_alpha_tuning = 0.05,
  store_payload = TRUE
) {

  performance_metric = match.arg(performance_metric)
  val_test = match.arg(val_test)

  if (!rs_outer$is_instantiated) {
    rs_outer$instantiate(task)
  }

  outer_iters = rs_outer$iters
  res_tbl = data.table::data.table()
  c_mats = vector("list", outer_iters)
  payloads = if (store_payload) vector("list", outer_iters) else NULL
  inner_scores = rep(NA_real_, outer_iters)

  for (i in seq_len(outer_iters)) {
    lgr$info(paste0("Starting outer fold ", i, "/", outer_iters, "..."))
    tr_idx = rs_outer$train_set(i)
    te_idx = rs_outer$test_set(i)
    task_tr = task$clone()$filter(tr_idx)
    task_te = task$clone()$filter(te_idx)

    gl_tune = graphlearner$clone(deep = TRUE)
    mbspls_id_tune = .mbspls_pipeop_id(gl_tune$graph, where = "gl_tune$graph")
    po_tune = gl_tune$graph$pipeops[[mbspls_id_tune]]
    po_tune$param_set$values$ncomp = as.integer(ncomp)
    po_tune$param_set$values$performance_metric = performance_metric

    tuner = TunerSeqMBsPLS$new(
      tuner              = "random_search",
      budget             = tuner_budget,
      resampling         = rs_inner,
      parallel           = "none",
      early_stopping     = tuning_early_stop,
      n_perm             = n_perm_tuning,
      perm_alpha         = perm_alpha_tuning,
      performance_metric = performance_metric
    )

    inst = mlr3tuning::ti(
      task        = task_tr,
      learner     = gl_tune,
      resampling  = rsmp("holdout"),
      measure     = msr("mbspls.mac_evwt"),
      terminator  = bbotk::trm("evals", n_evals = 1)
    )

    tuner$optimize(inst)

    c_star = inst$result$learner_param_vals[[1]]$c_matrix
    c_mats[[i]] = c_star

    lgr$info(paste0("  Finished tuning on outer fold ", i, "/", outer_iters, "."))

    inner_scores[i] = tryCatch(
      {
        as.numeric(inst$objective_result_y %||% inst$result_y %||%
          inst$archive$best()$y)
      },
      error = function(e) NA_real_
    )

    gl_eval = graphlearner$clone(deep = TRUE)
    mbspls_id_eval = .mbspls_pipeop_id(gl_eval$graph, where = "gl_eval$graph")
    po_eval = gl_eval$graph$pipeops[[mbspls_id_eval]]
    po_eval$param_set$values$ncomp = if (is.null(c_star)) as.integer(ncomp) else ncol(c_star)
    po_eval$param_set$values$performance_metric = performance_metric
    po_eval$param_set$values$c_matrix = c_star
    po_eval$param_set$values$permutation_test = FALSE
    po_eval$param_set$values$val_test = val_test
    po_eval$param_set$values$val_test_n = as.integer(val_test_n)
    po_eval$param_set$values$val_test_alpha = val_test_alpha
    po_eval$param_set$values$val_test_permute_all = isTRUE(val_permute_all)

    log_env_te = new.env(parent = emptyenv())
    po_eval$param_set$values$log_env = log_env_te

    lgr$info(paste0("  Evaluating on outer test fold ", i, "/", outer_iters, "..."))

    gl_eval$train(task_tr)
    gl_eval$predict(task_te)
    payload = log_env_te$last
    res_row = mbspls_nested_cv_result_row(
      split_id = i,
      inner_score = inner_scores[i],
      payload = payload,
      performance_metric = performance_metric,
      c_star = c_star
    )

    res_tbl = data.table::rbindlist(list(res_tbl, res_row), use.names = TRUE, fill = TRUE)
    if (store_payload) payloads[[i]] <- payload

    lgr$info(paste0("  Completed outer fold ", i, "/", outer_iters, "."))
  }

  summary_table = data.table::rbindlist(list(
    mbspls_metric_summary_row("MAC (LV1)", res_tbl$mac_lv1_test),
    mbspls_metric_summary_row("EV-weighted MAC (all LCs)", res_tbl$mac_evwt_test),
    mbspls_metric_summary_row("# components retained", res_tbl$ncomp_kept)
  ), use.names = TRUE, fill = TRUE)

  out = list(
    results       = res_tbl[],
    c_mats        = c_mats,
    inner_scores  = inner_scores,
    summary_table = summary_table
  )
  if (store_payload) out$payloads <- payloads
  out
}
