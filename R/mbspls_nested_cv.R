# Internal helpers shared by the direct and batchtools nested-CV paths.
mbspls_nested_cv_resolve_measure = function(measure) {
  if (is.character(measure)) {
    if (length(measure) != 1L) {
      stop("`measure` must be a single MB-sPLS measure id or an mlr3 Measure.", call. = FALSE)
    }
    measure = mlr3::msr(measure)
  }

  key = .mbspls_measure_key(measure)
  if (is.null(key)) {
    stop(
      "`measure` must be one of the package MB-sPLS measures: mbspls.mac_evwt, mbspls.mac, mbspls.ev, mbspls.block_ev.",
      call. = FALSE
    )
  }

  list(
    measure = measure,
    key = key,
    id = measure$id %||% key
  )
}

mbspls_nested_cv_primary_metric_label = function(measure_spec) {
  if (identical(measure_spec$key, "mbspls.mac_evwt")) {
    return("EV-weighted MAC (all LCs)")
  }
  sprintf("Test score (%s)", measure_spec$id)
}

mbspls_nested_cv_result_row = function(split_id, inner_score, payload, performance_metric, measure_spec, c_star = NULL, status = NULL) {
  missing_payload_status = status %||% sprintf(
    "Prediction payload is missing; measure '%s' could not be computed.",
    measure_spec$id
  )
  if (is.null(payload)) {
    return(data.table::data.table(
      split = split_id,
      measure_id = measure_spec$id,
      measure_key = measure_spec$key,
      inner_score = inner_score,
      measure_test = NA_real_,
      measure_test_defined = FALSE,
      measure_test_status = missing_payload_status,
      measure_test_n_undefined = NA_integer_,
      mac_lv1_test = NA_real_,
      mac_evwt_test = NA_real_,
      mac_evwt_defined = FALSE,
      mac_evwt_status = status %||% "Prediction payload is missing; measure 'mbspls.mac_evwt' could not be computed.",
      ncomp_kept = if (is.null(c_star)) NA_integer_ else ncol(c_star),
      perf_metric = performance_metric,
      val_p_last = NA_real_,
      val_p_all = list(NULL),
      val_stat_all = list(NULL)
    ))
  }

  mac = as.numeric(payload$mac_comp %||% numeric())
  vp = payload$val_test_p %||% payload$val_perm_p
  vs = payload$val_test_stat %||% NULL
  measure_info = mbspls_measure_score_diagnostics(payload, measure_spec$measure)
  mac_evwt_info = if (identical(measure_spec$key, "mbspls.mac_evwt")) {
    measure_info
  } else {
    mbspls_measure_score_diagnostics(payload, "mbspls.mac_evwt")
  }

  data.table::data.table(
    split = split_id,
    measure_id = measure_spec$id,
    measure_key = measure_spec$key,
    inner_score = inner_score,
    measure_test = measure_info$score,
    measure_test_defined = isTRUE(measure_info$defined),
    measure_test_status = if (isTRUE(measure_info$defined)) NA_character_ else as.character(measure_info$message %||% sprintf("Measure '%s' returned a non-finite score.", measure_spec$id)),
    measure_test_n_undefined = as.integer(measure_info$n_undefined_components %||% NA_integer_),
    mac_lv1_test = if (length(mac)) mac[1L] else NA_real_,
    mac_evwt_test = mac_evwt_info$score,
    mac_evwt_defined = isTRUE(mac_evwt_info$defined),
    mac_evwt_status = if (isTRUE(mac_evwt_info$defined)) NA_character_ else as.character(mac_evwt_info$message %||% "Measure 'mbspls.mac_evwt' returned a non-finite score."),
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

mbspls_nested_cv_summary_table = function(results, measure_spec) {
  summary_rows = list(
    mbspls_metric_summary_row(mbspls_nested_cv_primary_metric_label(measure_spec), results$measure_test),
    mbspls_metric_summary_row("MAC (LV1)", results$mac_lv1_test)
  )
  if (!identical(measure_spec$key, "mbspls.mac_evwt")) {
    summary_rows[[length(summary_rows) + 1L]] = mbspls_metric_summary_row("EV-weighted MAC (all LCs)", results$mac_evwt_test)
  }
  summary_rows[[length(summary_rows) + 1L]] = mbspls_metric_summary_row("# components retained", results$ncomp_kept)
  data.table::rbindlist(summary_rows, use.names = TRUE, fill = TRUE)
}

# Deep-clone `graphlearner` for one outer fold, set parameters of its MB-sPLS
# node, and give every PipeOp with a `log_env` parameter (MB-sPLS, bootstrap
# selection, ...) one fresh environment. The fold's nodes then share only their
# own state, and the environment of the supplied learner is never read or
# written.
.mbspls_nested_cv_learner = function(graphlearner, mbspls_values) {
  learner = graphlearner$clone(deep = TRUE)
  mbspls_id = .mbspls_pipeop_id(learner$graph, where = "graphlearner$graph")
  po_mbspls = learner$graph$pipeops[[mbspls_id]]
  for (id in names(mbspls_values)) {
    po_mbspls$param_set$values[[id]] = mbspls_values[[id]]
  }

  log_env = new.env(parent = emptyenv())
  log_env$warn_overwrite = FALSE
  for (node in learner$graph$pipeops) {
    if ("log_env" %in% node$param_set$ids()) {
      node$param_set$values$log_env = log_env
    }
  }
  list(learner = learner, log_env = log_env, pipeop_id = mbspls_id)
}

# Tune and evaluate one outer fold. Shared by mbspls_nested_cv() and the
# batchtools job, so both paths run identical code.
.mbspls_nested_cv_outer_fold = function(
  task, train_idx, test_idx, split_id, graphlearner, rs_inner, ncomp,
  tuner_budget, tuning_early_stop, measure_spec, performance_metric,
  val_test, val_test_n, val_test_alpha, val_permute_all, n_perm_tuning,
  perm_alpha_tuning
) {
  mb_assert_resampling_split(task, train_idx, test_idx)
  task_tr = task$clone()$filter(train_idx)
  task_te = task$clone()$filter(test_idx)

  tune = .mbspls_nested_cv_learner(graphlearner, list(
    ncomp = as.integer(ncomp),
    performance_metric = performance_metric
  ))
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
  # The sequential tuner runs its own inner resampling; the instance's
  # resampling and terminator are required by mlr3tuning but not used.
  inst = mlr3tuning::ti(
    task        = task_tr,
    learner     = tune$learner,
    resampling  = mlr3::rsmp("holdout"),
    measure     = measure_spec$measure,
    terminator  = bbotk::trm("evals", n_evals = 1L)
  )
  tuner$optimize(inst)
  c_star = inst$result$learner_param_vals[[1L]]$c_matrix
  inner_score = as.numeric(inst$result_y)

  evaluation = .mbspls_nested_cv_learner(graphlearner, list(
    ncomp = ncol(c_star),
    performance_metric = performance_metric,
    c_matrix = c_star,
    permutation_test = FALSE,
    val_test = val_test,
    val_test_n = as.integer(val_test_n),
    val_test_alpha = val_test_alpha,
    val_test_permute_all = isTRUE(val_permute_all)
  ))
  evaluation$learner$train(task_tr)
  evaluation$learner$predict(task_te)
  payload = .mb_prediction_payload(evaluation$learner, evaluation$pipeop_id)

  list(
    result_row = mbspls_nested_cv_result_row(
      split_id = split_id,
      inner_score = inner_score,
      payload = payload,
      performance_metric = performance_metric,
      measure_spec = measure_spec,
      c_star = c_star
    ),
    c_star = c_star,
    inner_score = inner_score,
    payload = payload
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
#' @details
#' Each outer fold tunes a deep clone of `graphlearner` with [TunerSeqMBsPLS] on
#' the analysis rows and then trains a second clone with the tuned `c_matrix`
#' on the same rows and predicts the assessment rows. Every PipeOp of these
#' clones that has a `log_env` parameter (the MB-sPLS node and, for example, a
#' [PipeOpMBsPLSBootstrapSelect] node) receives one fresh environment per fold,
#' so bootstrap stability selection runs on the fold's own MB-sPLS fit and the
#' outer estimate covers the complete configured pipeline. The supplied learner
#' and its log environment are left unchanged.
#'
#' `inner_scores` are the tuner's inner-resampling scores of the selected
#' `c_matrix`. They were maximised during tuning and are therefore optimistic;
#' the outer `measure_test` values are the performance estimate.
#'
#' If the task has an `mlr3` `group` column role, every outer split is checked
#' before fitting and the function errors if an exchangeability group occurs in
#' both analysis and assessment rows. The correct grouping variable remains a
#' study-design responsibility and cannot be inferred from feature values.
#'
#' `performance_metric` and `measure` operate at different layers of the
#' procedure.
#'
#' `performance_metric` selects the association criterion evaluated for
#' convergence and scoring (`"mac"` or `"frobenius"`). Weight updates follow
#' the same covariance-style sparse multiblock procedure for either criterion;
#' the solver does not directly optimize a correlation-gradient objective.
#'
#' `measure` controls model selection across candidate sparsity settings and
#' across resampling folds on held-out data. It therefore acts as an indirect
#' outer optimization criterion for tuning and evaluation.
#'
#' Choosing `measure = "mbspls.ev"` or `measure = "mbspls.block_ev"` does not
#' make the underlying MB-sPLS algorithm optimize explained variance directly.
#' It selects among fitted models using an EV-based validation criterion.
#'
#' For most analyses, `performance_metric = "mac"` with
#' `measure = "mbspls.mac_evwt"` remains the most natural default because it
#' combines direct latent-correlation fitting with held-out selection that also
#' downweights components lacking positive prediction-side explained variance.
#'
#' @param task [mlr3::Task] with all pre-MBsPLS features.
#' @param graphlearner [mlr3pipelines::GraphLearner] with a PipeOpMBsPLS node.
#' @param rs_outer [mlr3::Resampling] instantiated (outer) resampling.
#' @param rs_inner [mlr3::Resampling] template (inner) resampling.
#' @param ncomp integer, max #components considered by the tuner.
#' @param tuner_budget integer, #candidate evaluations per component.
#' @param tuning_early_stop logical, whether the tuner applies its
#'   permutation-based component stopping heuristic. The heuristic is evaluated
#'   on the inner validation folds that selected `c` and pools dependent folds,
#'   so it is optimistic; see [TunerSeqMBsPLS].
#' @param performance_metric "mac" or "frobenius" for the latent correlation.
#' @param measure [mlr3::Measure] or character(1). The package MB-sPLS measure
#'   used for inner tuning and aligned outer-fold scoring. Must resolve to one of
#'   `mbspls.mac_evwt`, `mbspls.mac`, `mbspls.ev`, or `mbspls.block_ev`.
#'   `mbspls.mac_evwt` remains the default for backward compatibility.
#' @param val_test "none", conditional "permutation", or descriptive
#'   "bootstrap" diagnostic run on the outer assessment split.
#' @param val_test_n integer, permutations/boot reps on OUTER test.
#' @param val_test_alpha numeric alpha for the descriptive bootstrap interval;
#'   retained for permutation API compatibility.
#' @param val_permute_all logical, permute all blocks in test validation.
#' @param n_perm_tuning integer, permutations used inside the tuner's early stop.
#' @param perm_alpha_tuning numeric, cutoff of the tuner's stopping heuristic
#'   (component-wise). It is not an error rate.
#' @param store_payload logical, keep the full PipeOpMBsPLS predict payload.
#'
#' @return list with:
#'   - results: data.table with per-outer-split summary, including the selected
#'     objective (`measure_id`, `measure_key`, `measure_test`,
#'     `measure_test_defined`, `measure_test_status`), the number of components
#'     whose latent correlation was undefined on the assessment rows and scored
#'     as zero (`measure_test_n_undefined`), plus secondary diagnostics such as
#'     `mac_lv1_test` and `mac_evwt_test`
#'   - c_mats: list of tuned C* matrices per outer split
#'   - inner_scores: numeric vector of the tuner's inner-CV scores of the
#'     selected C* (one per outer split; optimistic)
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
  measure = mlr3::msr("mbspls.mac_evwt"),
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
  measure_spec = mbspls_nested_cv_resolve_measure(measure)

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
    fold = .mbspls_nested_cv_outer_fold(
      task = task,
      train_idx = rs_outer$train_set(i),
      test_idx = rs_outer$test_set(i),
      split_id = i,
      graphlearner = graphlearner,
      rs_inner = rs_inner,
      ncomp = ncomp,
      tuner_budget = tuner_budget,
      tuning_early_stop = tuning_early_stop,
      measure_spec = measure_spec,
      performance_metric = performance_metric,
      val_test = val_test,
      val_test_n = val_test_n,
      val_test_alpha = val_test_alpha,
      val_permute_all = val_permute_all,
      n_perm_tuning = n_perm_tuning,
      perm_alpha_tuning = perm_alpha_tuning
    )

    c_mats[i] = list(fold$c_star)
    inner_scores[i] = fold$inner_score
    res_tbl = data.table::rbindlist(list(res_tbl, fold$result_row), use.names = TRUE, fill = TRUE)
    if (store_payload) {
      payloads[i] = list(fold$payload)
    }

    lgr$info(paste0("  Completed outer fold ", i, "/", outer_iters, "."))
  }

  summary_table = mbspls_nested_cv_summary_table(res_tbl, measure_spec)

  out = list(
    results       = res_tbl[],
    c_mats        = c_mats,
    inner_scores  = inner_scores,
    summary_table = summary_table,
    measure_id    = measure_spec$id,
    measure_key   = measure_spec$key
  )
  if (store_payload) {
    out$payloads = payloads
  }
  out
}
