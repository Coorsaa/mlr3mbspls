# -----------------------------------------------------------------------------
# Batchtools: run each OUTER fold of mbspls_nested_cv() as a separate job
# -----------------------------------------------------------------------------

# Internal job function (one OUTER fold). It runs the same code as one
# iteration of mbspls_nested_cv().
.mbspls_outer_job = function(
  train_idx, test_idx, split_id,
  task, graphlearner, rs_inner,
  ncomp, tuner_budget, tuning_early_stop,
  measure,
  performance_metric, val_test, val_test_n, val_test_alpha,
  val_permute_all, n_perm_tuning, perm_alpha_tuning,
  store_payload
) {
  fold = .mbspls_nested_cv_outer_fold(
    task = task,
    train_idx = train_idx,
    test_idx = test_idx,
    split_id = split_id,
    graphlearner = graphlearner,
    rs_inner = rs_inner,
    ncomp = ncomp,
    tuner_budget = tuner_budget,
    tuning_early_stop = tuning_early_stop,
    measure_spec = mbspls_nested_cv_resolve_measure(measure),
    performance_metric = performance_metric,
    val_test = val_test,
    val_test_n = val_test_n,
    val_test_alpha = val_test_alpha,
    val_permute_all = val_permute_all,
    n_perm_tuning = n_perm_tuning,
    perm_alpha_tuning = perm_alpha_tuning
  )
  if (!isTRUE(store_payload)) {
    fold["payload"] = list(NULL)
  }
  fold
}

# Unwrap the list returned by mbspls_nested_cv_batchtools().
.mbspls_batchtools_registry = function(reg) {
  if (!inherits(reg, "Registry") && is.list(reg) && inherits(reg$reg, "Registry")) {
    reg = reg$reg
  }
  checkmate::assert_class(reg, "Registry", .var.name = "reg")
  reg
}


#' Perform nested cross-validation for MB-sPLS using batchtools
#' @description
#' This function performs nested cross-validation for MB-sPLS using the
#' `batchtools` package to parallelize the outer folds as separate jobs.
#' Each outer fold is processed in a separate job, which tunes the MB-sPLS
#' model on the inner folds and evaluates it on the outer test fold.
#' The results are collected and summarized with [collect_mbspls_nested_cv()]
#' after the jobs have completed.
#'
#' @details
#' Each job runs exactly the code of one outer fold of [mbspls_nested_cv()]:
#' the tuner and the evaluation learner are deep clones of `graphlearner`, and
#' every PipeOp with a `log_env` parameter (the MB-sPLS node and, for example,
#' a [PipeOpMBsPLSBootstrapSelect] node) receives one fresh environment per job.
#' Graphs with bootstrap stability selection are therefore evaluated as
#' configured.
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
#' @param task An `mlr3` task (supervised or unsupervised).
#' @param graphlearner An `mlr3` `GraphLearner` that implements MB-sPLS
#'   (e.g., created with `mbspls_graph_learner()`).
#' @param rs_outer An `mlr3` resampling instance for the outer folds.
#' @param rs_inner An `mlr3` resampling template for the inner folds; it is
#'   instantiated on the analysis rows of every outer fold.
#' @param ncomp Integer, maximum number of components to consider.
#' @param tuner_budget Integer, maximum number of evaluations for the tuner.
#' @param tuning_early_stop Logical, whether the tuner applies its
#'   permutation-based component stopping heuristic. The heuristic is
#'   evaluated on the inner validation folds that selected `c` and pools
#'   dependent folds, so it is optimistic; see [TunerSeqMBsPLS].
#' @param measure [mlr3::Measure] or character(1). The package MB-sPLS measure
#'   used for inner tuning and aligned outer-fold scoring. Must resolve to one of
#'   `mbspls.mac_evwt`, `mbspls.mac`, `mbspls.ev`, or `mbspls.block_ev`.
#'   `mbspls.mac_evwt` remains the default for backward compatibility.
#' @param performance_metric Character, performance metric to optimize
#'   ("mac" or "frobenius").
#' @param val_test Character, prediction-side diagnostic: "none", conditional
#'   "permutation", or descriptive "bootstrap".
#' @param val_test_n Integer, number of permutations or bootstrap samples
#'   for the prediction-side diagnostic (default is 1000).
#' @param val_test_alpha Numeric alpha for the descriptive bootstrap interval;
#'   retained for permutation API compatibility (default is 0.05).
#' @param val_permute_all Logical, whether to permute all blocks during
#'   the conditional permutation diagnostic (default is TRUE).
#' @param n_perm_tuning Integer, number of permutations for the tuning
#'   stopping heuristic (default is 500).
#' @param perm_alpha_tuning Numeric cutoff of the tuning stopping heuristic
#'   (default is 0.05). It is not an error rate.
#' @param store_payload Logical, whether to store the full payload from
#'   each outer fold (may consume a lot of memory).
#' @param reg_dir Character, directory to store the `batchtools` registry.
#' @param seed Integer, random seed for reproducibility.
#' @param cluster_function A `batchtools` cluster function object, or `NULL`
#'   (default) for a local socket setup created via
#'   `batchtools::makeClusterFunctionsSocket(ncpus = 1L)`.
#' @param autosubmit Logical, whether to automatically submit the jobs
#'   after creating the registry (default is FALSE).
#' @return A list with elements `ids`, the job table returned by
#'   `batchtools::batchMap()` (one job per outer split), and `reg`, the
#'   `batchtools` Registry. Submit the jobs with
#'   `batchtools::submitJobs(reg = out$reg)` unless `autosubmit = TRUE`, and pass
#'   `reg = out$reg` (or the whole list) to [collect_mbspls_nested_cv()].
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
  cluster_function = NULL,
  autosubmit = FALSE
) {
  .mbspls_require_suggested("batchtools", "mbspls_nested_cv_batchtools()")
  measure_spec = mbspls_nested_cv_resolve_measure(measure)
  performance_metric = match.arg(performance_metric)
  val_test = match.arg(val_test)
  cluster_function = cluster_function %||% batchtools::makeClusterFunctionsSocket(ncpus = 1L)

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

  list(ids = ids, reg = reg)
}


#' Collect and summarize results from MB-sPLS nested CV Batchtools jobs
#' @description This function collects and summarizes the results from the
#'   MB-sPLS nested cross-validation jobs that were submitted to the
#'   Batchtools registry. It gathers the results from each outer fold,
#'   including the final model performance metrics and the inner cross-validation
#'   results. It also computes summary statistics across all outer folds.
#'
#' @details
#' An outer-performance estimate computed from the folds that happened to
#' finish is biased and would be presented as complete, so by default every
#' requested job must have finished successfully. If a job errored, expired,
#' is still queued or running, or was never submitted, the function stops with
#' a report of the affected outer splits. With `allow_partial = TRUE` it warns
#' instead and adds a row with missing scores for each unfinished split, whose
#' `measure_test_status` describes the job status (including the error
#' message). These rows count towards `n_total` and `n_failed` in the summary
#' table.
#'
#' @param ids A vector (or job table) of job IDs of the outer fold jobs to
#'   collect, or `NULL` (default) for all jobs of the registry. IDs that are not
#'   in the registry are an error.
#' @param reg A `batchtools` registry object, or the list returned by
#'   [mbspls_nested_cv_batchtools()].
#' @param allow_partial Logical; if `TRUE`, summarise the finished jobs and
#'   report unfinished or failed jobs as rows with missing scores instead of
#'   stopping (default `FALSE`).
#' @return A list containing:
#'   - `results`: A data.table with one row per requested outer split, ordered
#'     by split, including the selected objective (`measure_id`, `measure_key`,
#'     `measure_test`, `measure_test_defined`, `measure_test_status`), the
#'     number of components whose latent correlation was undefined and scored as
#'     zero (`measure_test_n_undefined`), plus secondary diagnostics such as
#'     `mac_lv1_test` and `mac_evwt_test`.
#'   - `c_mats`: A list of the tuned C* matrices from each outer split (`NULL`
#'     for unfinished splits).
#'   - `inner_scores`: A numeric vector of the tuner's inner cross-validation
#'     scores from each outer split (`NA` for unfinished splits).
#'   - `payloads`: (optional) A list of the full payloads from each outer fold
#'     (if `store_payload = TRUE` was set).
#'   - `summary_table`: A data.table with summary statistics across all outer folds,
#'     including `n_total`, `n_defined`, and `n_failed`.
#' @seealso `mbspls_nested_cv_batchtools()`
#' @import data.table
#' @importFrom lgr lgr
#' @export
collect_mbspls_nested_cv = function(ids = NULL, reg, allow_partial = FALSE) {
  .mbspls_require_suggested("batchtools", "collect_mbspls_nested_cv()")
  reg = .mbspls_batchtools_registry(reg)
  checkmate::assert_flag(allow_partial)

  all_ids = batchtools::findJobs(reg = reg)$job.id
  if (is.null(ids)) {
    ids = all_ids
  } else {
    if (is.data.frame(ids)) {
      ids = ids$job.id
    }
    checkmate::assert_integerish(ids, any.missing = FALSE, min.len = 1L)
    ids = unique(as.integer(ids))
    unknown = setdiff(ids, all_ids)
    if (length(unknown)) {
      stop(sprintf(
        "Job id(s) not found in the registry: %s.",
        paste(unknown, collapse = ", ")
      ), call. = FALSE)
    }
  }
  if (!length(ids)) {
    stop("The registry contains no nested-CV jobs to collect.", call. = FALSE)
  }
  done = intersect(ids, batchtools::findDone(ids = ids, reg = reg)$job.id)
  missing = setdiff(ids, done)
  job_pars = batchtools::getJobPars(ids = ids, reg = reg)
  split_of = stats::setNames(
    vapply(job_pars$job.pars, function(p) as.integer(p$split_id), integer(1L)),
    job_pars$job.id
  )

  missing_status = character()
  if (length(missing)) {
    errors = intersect(missing, batchtools::findErrors(ids = missing, reg = reg)$job.id)
    expired = intersect(missing, batchtools::findExpired(ids = missing, reg = reg)$job.id)
    missing_status = stats::setNames(rep("Outer fold job has not finished (not submitted, queued or running).", length(missing)), missing)
    if (length(expired)) {
      missing_status[as.character(expired)] = "Outer fold job expired before finishing."
    }
    if (length(errors)) {
      messages = batchtools::getErrorMessages(ids = errors, reg = reg)
      missing_status[as.character(messages$job.id)] = sprintf("Outer fold job failed: %s", messages$message)
    }
    report = sprintf(
      "%d of %d outer fold jobs did not complete (errors: %d, expired: %d, not finished: %d; outer splits %s)",
      length(missing), length(ids), length(errors), length(expired),
      length(missing) - length(errors) - length(expired),
      paste(sort(split_of[as.character(missing)]), collapse = ", ")
    )
    if (!allow_partial) {
      stop(report, ". Rerun or finish these jobs, or set allow_partial = TRUE to summarise the completed folds only.", call. = FALSE)
    }
    warning(report, ". Their rows have missing scores and count as failed in the summary.", call. = FALSE)
  }

  # each finished job: list(result_row, c_star, inner_score, payload)
  res_list = if (length(done)) batchtools::reduceResultsList(ids = done, reg = reg) else list()
  names(res_list) = as.character(done)
  lgr$info("Collecting results from %d of %d outer folds...", length(done), length(ids))

  done_rows = data.table::rbindlist(lapply(res_list, `[[`, "result_row"), use.names = TRUE, fill = TRUE)
  measure_ids = unique(stats::na.omit(done_rows$measure_id %||% NA_character_))
  measure_keys = unique(stats::na.omit(done_rows$measure_key %||% NA_character_))
  if (length(measure_ids) > 1L || length(measure_keys) > 1L) {
    stop("collect_mbspls_nested_cv() expects a single tuning measure per registry.", call. = FALSE)
  }
  # The objective is part of every job's registered arguments, so rows of
  # unfinished jobs are labelled correctly even when no job has finished.
  registered = batchtools::makeJob(ids[[1L]], reg = reg)$pars
  measure_spec = mbspls_nested_cv_resolve_measure(registered$measure)
  if ((length(measure_keys) && !identical(measure_keys[[1L]], measure_spec$key)) ||
    (length(measure_ids) && !identical(measure_ids[[1L]], measure_spec$id))) {
    stop("Finished jobs report a tuning measure that differs from the registered one.", call. = FALSE)
  }
  perf_metrics = unique(stats::na.omit(done_rows$perf_metric %||% NA_character_))
  perf_metric = if (length(perf_metrics) == 1L) {
    perf_metrics[[1L]]
  } else if (!length(perf_metrics) && checkmate::test_string(registered$performance_metric)) {
    registered$performance_metric
  } else {
    NA_character_
  }

  # One entry per requested outer split, ordered by split.
  ordered_ids = as.character(ids[order(split_of[as.character(ids)])])
  rows = lapply(ordered_ids, function(id) {
    if (id %in% names(res_list)) {
      return(res_list[[id]]$result_row)
    }
    mbspls_nested_cv_result_row(
      split_id = split_of[[id]],
      inner_score = NA_real_,
      payload = NULL,
      performance_metric = perf_metric,
      measure_spec = measure_spec,
      status = missing_status[[id]]
    )
  })
  results = data.table::rbindlist(rows, use.names = TRUE, fill = TRUE)
  c_mats = lapply(ordered_ids, function(id) res_list[[id]]$c_star)
  inner_scores = vapply(ordered_ids, function(id) {
    as.numeric(res_list[[id]]$inner_score %||% NA_real_)
  }, numeric(1L), USE.NAMES = FALSE)
  payloads = lapply(ordered_ids, function(id) res_list[[id]]$payload)

  out = list(
    results       = results[],
    c_mats        = c_mats,
    inner_scores  = inner_scores,
    summary_table = mbspls_nested_cv_summary_table(results, measure_spec),
    measure_id    = measure_spec$id,
    measure_key   = measure_spec$key
  )
  # only attach payloads if they were produced
  if (any(vapply(payloads, Negate(is.null), logical(1L)))) {
    out$payloads = payloads
  }
  out
}
