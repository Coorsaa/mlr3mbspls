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
#'   - results: data.table with per-outer-split summary (inner_score, test MACs, etc.)
#'   - c_mats: list of tuned C* matrices per outer split
#'   - inner_scores: numeric vector of best inner-CV scores (one per outer split)
#'   - payloads: (optional) list of PipeOp logging payloads for each outer test
#'   - summary_table: data.table with aggregated metrics across outer splits
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
      learner     = graphlearner,
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

    po = graphlearner$graph$pipeops$mbspls
    po$param_set$values$c_matrix = c_star
    po$param_set$values$permutation_test = TRUE
    po$param_set$values$val_test = val_test
    po$param_set$values$val_test_n = as.integer(val_test_n)
    po$param_set$values$val_test_alpha = val_test_alpha
    po$param_set$values$val_test_permute_all = isTRUE(val_permute_all)

    log_env_te = new.env(parent = emptyenv())
    po$param_set$values$log_env = log_env_te

    lgr$info(paste0("  Evaluating on outer test fold ", i, "/", outer_iters, "..."))

    graphlearner$train(task_tr)
    graphlearner$predict(task_te)
    payload = log_env_te$last
    if (is.null(payload)) {
      res_row = data.table::data.table(
        split          = i,
        inner_score    = inner_scores[i],
        mac_lv1_test   = NA_real_,
        mac_evwt_test  = NA_real_,
        ncomp_kept     = if (is.null(c_star)) NA_integer_ else ncol(c_star),
        perf_metric    = performance_metric,
        val_p_last     = NA_real_
      )
    } else {
      mac = as.numeric(payload$mac_comp %||% NA_real_)
      ev = as.numeric(payload$ev_comp %||% NA_real_)
      w = ev / (sum(ev) + 1e-12)
      vp = payload$val_test_p %||% payload$val_perm_p # backward compat
      vs = payload$val_test_stat %||% NULL
      res_row = data.table::data.table(
        split          = i,
        inner_score    = inner_scores[i],
        mac_lv1_test   = mac[1],
        mac_evwt_test  = sum(w * mac),
        ncomp_kept     = length(mac),
        perf_metric    = payload$perf_metric %||% performance_metric,
        val_p_last     = if (!is.null(vp)) as.numeric(utils::tail(vp, 1L)) else NA_real_,
        val_p_all      = list(vp),
        val_stat_all   = list(vs)
      )
    }

    res_tbl = data.table::rbindlist(list(res_tbl, res_row), use.names = TRUE, fill = TRUE)
    if (store_payload) payloads[[i]] <- payload

    lgr$info(paste0("  Completed outer fold ", i, "/", outer_iters, "."))
  }

  # --- pretty summary table across outer splits (no .SD/.SDcols tricks) -----
  .summ = function(x) {
    x = as.numeric(x)
    data.table::data.table(
      mean   = mean(x, na.rm = TRUE),
      sd     = stats::sd(x, na.rm = TRUE),
      median = stats::median(x, na.rm = TRUE),
      q05    = as.numeric(stats::quantile(x, 0.05, na.rm = TRUE)),
      q95    = as.numeric(stats::quantile(x, 0.95, na.rm = TRUE)),
      min    = suppressWarnings(min(x, na.rm = TRUE)),
      max    = suppressWarnings(max(x, na.rm = TRUE)),
      n      = sum(is.finite(x))
    )
  }

  row = function(label, x) {
    cbind(
      data.table::data.table(metric = label),
      .summ(x)
    )
  }

  summary_table = data.table::rbindlist(list(
    row("MAC (LV1)", res_tbl$mac_lv1_test),
    row("EV-weighted MAC (all LCs)", res_tbl$mac_evwt_test),
    row("# components retained", res_tbl$ncomp_kept)
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
