# -----------------------------------------------------------------------------
# Batchtools: run each OUTER fold of mbspls_nested_cv() as a separate job
# -----------------------------------------------------------------------------

.mbspls_nested_cv_resolve_measure_batch = function(measure) {
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


.mbspls_nested_cv_primary_metric_label_batch = function(measure_spec) {
  if (identical(measure_spec$key, "mbspls.mac_evwt")) {
    return("EV-weighted MAC (all LCs)")
  }
  sprintf("Test score (%s)", measure_spec$id)
}

.mbspls_nested_cv_result_row_batch = function(split_id, inner_score, payload, performance_metric, measure_spec, c_star = NULL) {
  missing_payload_status = sprintf(
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
      mac_lv1_test = NA_real_,
      mac_evwt_test = NA_real_,
      mac_evwt_defined = FALSE,
      mac_evwt_status = "Prediction payload is missing; measure 'mbspls.mac_evwt' could not be computed.",
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


.mbspls_metric_summary_batch = function(x) {
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


.mbspls_metric_summary_row_batch = function(label, x) {
  cbind(
    data.table::data.table(metric = label),
    .mbspls_metric_summary_batch(x)
  )
}

# internal job function (one OUTER fold)
.mbspls_outer_job = function(
  train_idx, test_idx, split_id,
  task, graphlearner, rs_inner,
  ncomp, tuner_budget, tuning_early_stop,
  measure,
  performance_metric, val_test, val_test_n, val_test_alpha,
  val_permute_all, n_perm_tuning, perm_alpha_tuning,
  store_payload
) {

  # shallow %||% (avoid importing rlang)
  or_null = function(x, y) if (is.null(x)) y else x
  measure_spec = .mbspls_nested_cv_resolve_measure_batch(measure)

  # clone task and GL to avoid state carry-over across jobs
  task_tr = task$clone()$filter(train_idx)
  task_te = task$clone()$filter(test_idx)

  gl_tune = graphlearner$clone(deep = TRUE)
  # ensure mbspls node reflects arguments (ncomp + metric)
  mbspls_id_tune = .mbspls_pipeop_id(gl_tune$graph, where = "gl_tune$graph")
  po_tune = gl_tune$graph$pipeops[[mbspls_id_tune]]
  po_tune$param_set$values$ncomp = ncomp
  po_tune$param_set$values$performance_metric = performance_metric

  # build sequential tuner (same as in your function)
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
    resampling  = mlr3::rsmp("holdout"), # ignored by our tuner except for naming
    measure     = measure_spec$measure,
    terminator  = bbotk::trm("evals", n_evals = 1)
  )
  tuner$optimize(inst)

  # tuned C*
  c_star = inst$result$learner_param_vals[[1]]$c_matrix

  # best inner score (robust extraction)
  inner_score = tryCatch(
    as.numeric(or_null(inst$objective_result_y,
      or_null(inst$result_y, inst$archive$best()$y))),
    error = function(e) NA_real_
  )

  # evaluation GraphLearner clone for outer test
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

  # prediction-side payload
  log_env_te = new.env(parent = emptyenv())
  po_eval$param_set$values$log_env = log_env_te

  gl_eval$train(task_tr)
  gl_eval$predict(task_te)
  payload = log_env_te$last

  # fold summary row (identical fields to your function)
  res_row = .mbspls_nested_cv_result_row_batch(
    split_id = split_id,
    inner_score = inner_score,
    payload = payload,
    performance_metric = performance_metric,
    measure_spec = measure_spec,
    c_star = c_star
  )

  list(
    result_row  = res_row,
    c_star      = c_star,
    inner_score = inner_score,
    payload     = if (isTRUE(store_payload)) payload else NULL
  )
}


#' Perform nested cross-validation for MB-sPLS using batchtools
#' @description
#' This function performs nested cross-validation for MB-sPLS using the
#' `batchtools` package to parallelize the outer folds as separate jobs.
#' Each outer fold is processed in a separate job, which tunes the MB-sPLS
#' model on the inner folds and evaluates it on the outer test fold.
#' The results are collected and summarized after all jobs are completed.
#'
#' @details
#' `performance_metric` and `measure` operate at different layers of the
#' procedure.
#'
#' `performance_metric` controls the direct objective optimized inside the
#' MB-sPLS C++ fitting routine when each latent component is estimated. It
#' therefore changes how the component weights are fitted (`"mac"` or
#' `"frobenius"`).
#'
#' `measure` controls model selection across candidate sparsity settings and
#' across resampling folds on held-out data. It therefore acts as an indirect
#' outer optimization criterion for tuning and evaluation.
#'
#' Choosing `measure = "mbspls.ev"` or `measure = "mbspls.block_ev"` does not
#' make the underlying MB-sPLS algorithm optimize explained variance directly.
#' It selects among models that were still fitted with the chosen
#' `performance_metric`, using an EV-based validation criterion.
#' @param task An `mlr3` task (supervised or unsupervised).
#' @param graphlearner An `mlr3` `GraphLearner` that implements MB-sPLS
#'   (e.g., created with `mbspls_graph_learner()`).
#' @param rs_outer An `mlr3` resampling instance for the outer folds.
#' @param rs_inner An `mlr3` resampling instance for the inner folds.
#' @param ncomp Integer, maximum number of components to consider.
#' @param tuner_budget Integer, maximum number of evaluations for the tuner.
#' @param tuning_early_stop Logical, whether to stop tuning early if
#'   non-significant components are found.
#' @param measure [mlr3::Measure] or character(1). The package MB-sPLS measure
#'   used for inner tuning and aligned outer-fold scoring. Must resolve to one of
#'   `mbspls.mac_evwt`, `mbspls.mac`, `mbspls.ev`, or `mbspls.block_ev`.
#'   `mbspls.mac_evwt` remains the default for backward compatibility.
#' @param performance_metric Character, performance metric to optimize
#'   ("mac" or "frobenius").
#' @param val_test Character, type of validation test to perform
#'   ("none", "permutation", or "bootstrap").
#' @param val_test_n Integer, number of permutations or bootstrap samples
#'   for the validation test (default is 1000).
#' @param val_test_alpha Numeric, significance level for the validation test
#'   (default is 0.05).
#' @param val_permute_all Logical, whether to permute all blocks during
#'   the validation test (default is TRUE).
#' @param n_perm_tuning Integer, number of permutations for the tuning
#'   significance test (default is 500).
#' @param perm_alpha_tuning Numeric, significance level for the tuning
#'   permutation test (default is 0.05).
#' @param store_payload Logical, whether to store the full payload from
#'   each outer fold (may consume a lot of memory).
#' @param reg_dir Character, directory to store the `batchtools` registry.
#' @param seed Integer, random seed for reproducibility.
#' @param cluster_function A `batchtools` cluster function object.
#'   Defaults to a local socket setup created via
#'   `batchtools::makeClusterFunctionsSocket()`.
#' @param autosubmit Logical, whether to automatically submit the jobs
#'   after creating the registry (default is FALSE).
#' @return A `batchtools` registry object. Use `collect_mbspls_nested_cv()`
#'   to gather and summarize the results after all jobs are completed.
#'
#' @export
mbspls_nested_cv_batchtools = function(
  task,
  graphlearner,
  rs_outer,
  rs_inner,
  ncomp,
  tuner_budget,
  tuning_early_stop = TRUE,
  measure = mlr3::msr("mbspls.mac_evwt"),
  performance_metric = c("mac", "frobenius"),
  val_test = c("none", "permutation", "bootstrap"),
  val_test_n = 1000L,
  val_test_alpha = 0.05,
  val_permute_all = TRUE,
  n_perm_tuning = 500L,
  perm_alpha_tuning = 0.05,
  store_payload = TRUE,
  reg_dir = "registry_mbspls_nestedcv",
  seed = 1L,
  cluster_function = batchtools::makeClusterFunctionsSocket(ncpus = 1L),
  autosubmit = FALSE
) {
  measure_spec = .mbspls_nested_cv_resolve_measure_batch(measure)
  performance_metric = match.arg(performance_metric)
  val_test = match.arg(val_test)

  # ensure OUTER is instantiated; we only pass indices to jobs
  if (!rs_outer$is_instantiated) rs_outer$instantiate(task)

  outer_iters = rs_outer$iters
  outer_sets = lapply(seq_len(outer_iters), function(i) {
    list(train = rs_outer$train_set(i), test = rs_outer$test_set(i), split = i)
  })

  # registry & CFs (socket clusters OK locally)
  lgr$info("Creating Batchtools registry...")
  reg = batchtools::makeRegistry(
    file.dir = reg_dir, seed = seed
  )
  reg$cluster.functions = cluster_function
  # map one job per OUTER split
  ids = batchtools::batchMap(
    fun = .mbspls_outer_job,
    train_idx = lapply(outer_sets, `[[`, "train"),
    test_idx = lapply(outer_sets, `[[`, "test"),
    split_id = vapply(outer_sets, `[[`, integer(1), "split"),
    more.args = list(
      task               = task,
      graphlearner       = graphlearner,
      rs_inner           = rs_inner,
      ncomp              = ncomp,
      tuner_budget       = tuner_budget,
      tuning_early_stop  = tuning_early_stop,
      measure            = measure_spec$measure,
      performance_metric = performance_metric,
      val_test           = val_test,
      val_test_n         = val_test_n,
      val_test_alpha     = val_test_alpha,
      val_permute_all    = val_permute_all,
      n_perm_tuning      = n_perm_tuning,
      perm_alpha_tuning  = perm_alpha_tuning,
      store_payload      = store_payload
    ),
    reg = reg
  )

  if (isTRUE(autosubmit)) {
    batchtools::submitJobs(reg = reg)
  }

  return(list(ids = ids, reg = invisible(reg)))
}


#' Collect and summarize results from MB-sPLS nested CV Batchtools jobs
#' @description This function collects and summarizes the results from the
#'   MB-sPLS nested cross-validation jobs that were submitted to the
#'   Batchtools registry. It gathers the results from each outer fold,
#'   including the final model performance metrics and the inner cross-validation
#'   results. It also computes summary statistics across all outer folds.
#' @param ids A vector of job IDs corresponding to the completed outer fold jobs.
#' @param reg A `batchtools` registry object returned by
#'   `mbspls_nested_cv_batchtools()`.
#' @return A list containing:
#'   - `results`: A data.table with the results from each outer fold, including
#'     the selected objective (`measure_id`, `measure_key`, `measure_test`,
#'     `measure_test_defined`, `measure_test_status`) plus secondary diagnostics
#'     such as `mac_lv1_test` and `mac_evwt_test`.
#'   - `c_mats`: A list of the tuned C* matrices from each outer fold.
#'   - `inner_scores`: A numeric vector of the best inner cross-validation
#'     scores from each outer fold.
#'   - `payloads`: (optional) A list of the full payloads from each outer fold
#'     (if `store_payload = TRUE` was set).
#'   - `summary_table`: A data.table with summary statistics across all outer folds,
#'     including `n_total`, `n_defined`, and `n_failed`.
#' @seealso `mbspls_nested_cv_batchtools()`
#' @import data.table
#' @importFrom lgr lgr
#' @export
collect_mbspls_nested_cv = function(ids = NULL, reg) {
  if (is.null(ids)) {
    ids = batchtools::findDone(reg = reg)$job.id
  }
  # gather per-job lists
  res_list = batchtools::reduceResultsList(ids = ids, reg = reg)
  # each item: list(result_row, c_star, inner_score, payload)

  lgr$info("Collecting results from %d outer folds...", length(res_list))

  # combine result rows
  results = data.table::rbindlist(lapply(res_list, `[[`, "result_row"),
    use.names = TRUE, fill = TRUE)
  c_mats = lapply(res_list, `[[`, "c_star")
  inner_scores = vapply(res_list, function(z) as.numeric(z$inner_score), 1.0)
  payloads = lapply(res_list, `[[`, "payload")
  measure_ids = unique(stats::na.omit(results$measure_id %||% NA_character_))
  measure_keys = unique(stats::na.omit(results$measure_key %||% NA_character_))
  if (length(measure_ids) > 1L || length(measure_keys) > 1L) {
    stop("collect_mbspls_nested_cv() expects a single tuning measure per registry.", call. = FALSE)
  }
  measure_spec = list(
    id = if (length(measure_ids)) measure_ids[[1L]] else "mbspls.mac_evwt",
    key = if (length(measure_keys)) measure_keys[[1L]] else "mbspls.mac_evwt"
  )

  # pretty summary table (same schema you used)
  summary_rows = list(
    .mbspls_metric_summary_row_batch(.mbspls_nested_cv_primary_metric_label_batch(measure_spec), results$measure_test),
    .mbspls_metric_summary_row_batch("MAC (LV1)", results$mac_lv1_test)
  )
  if (!identical(measure_spec$key, "mbspls.mac_evwt")) {
    summary_rows[[length(summary_rows) + 1L]] = .mbspls_metric_summary_row_batch("EV-weighted MAC (all LCs)", results$mac_evwt_test)
  }
  summary_rows[[length(summary_rows) + 1L]] = .mbspls_metric_summary_row_batch("# components retained", results$ncomp_kept)
  summary_table = data.table::rbindlist(summary_rows, use.names = TRUE, fill = TRUE)

  out = list(
    results       = results[],
    c_mats        = c_mats,
    inner_scores  = inner_scores,
    summary_table = summary_table,
    measure_id    = measure_spec$id,
    measure_key   = measure_spec$key
  )
  # only attach payloads if they were produced
  if (any(vapply(payloads, Negate(is.null), logical(1)))) out$payloads <- payloads
  out
}
