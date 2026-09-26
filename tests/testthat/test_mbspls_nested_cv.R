test_that("mbspls_nested_cv does not mutate the supplied GraphLearner", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 31L)
  gl = mbspls_graph_learner(
    learner = mlr3::lrn("clust.kmeans", centers = 2L),
    task = task,
    ncomp = 1L,
    bootstrap = FALSE,
    bootstrap_selection = FALSE,
    val_test = "none"
  )

  mbspls_id = .mbspls_pipeop_id(gl$graph, where = "gl$graph")
  expect_null(gl$model)
  expect_null(gl$graph$pipeops[[mbspls_id]]$param_set$values$c_matrix)

  res = NULL
  expect_no_error({
    res = mbspls_nested_cv(
      task = task,
      graphlearner = gl,
      rs_outer = mlr3::rsmp("holdout"),
      rs_inner = mlr3::rsmp("holdout"),
      ncomp = 1L,
      tuner_budget = 1L,
      tuning_early_stop = FALSE,
      val_test = "none",
      n_perm_tuning = 1L,
      store_payload = FALSE
    )
  })

  expect_true(all(c("measure_id", "measure_key", "measure_test", "measure_test_defined", "measure_test_status") %in% names(res$results)))
  expect_true(is.integer(res$results$measure_test_n_undefined))
  expect_identical(res$results$measure_test_n_undefined, 0L)
  expect_true(all(c("mac_evwt_defined", "mac_evwt_status") %in% names(res$results)))
  expect_true(all(c("n_total", "n_defined", "n_failed") %in% names(res$summary_table)))
  expect_equal(unique(res$summary_table$n_total), 1L)
  expect_identical(res$measure_key, "mbspls.mac_evwt")
  expect_identical(res$measure_id, "mbspls.mac_evwt")

  expect_null(gl$model)
  expect_null(gl$graph$pipeops[[mbspls_id]]$param_set$values$c_matrix)
})


test_that("mbspls_nested_cv supports package MB-sPLS measures beyond mac_evwt", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 32L)
  gl = mbspls_graph_learner(
    learner = mlr3::lrn("clust.kmeans", centers = 2L),
    task = task,
    ncomp = 1L,
    bootstrap = FALSE,
    bootstrap_selection = FALSE,
    val_test = "none"
  )

  res = NULL
  expect_no_error({
    res = mbspls_nested_cv(
      task = task,
      graphlearner = gl,
      rs_outer = mlr3::rsmp("holdout"),
      rs_inner = mlr3::rsmp("holdout"),
      ncomp = 1L,
      tuner_budget = 1L,
      tuning_early_stop = FALSE,
      measure = mlr3::msr("mbspls.ev"),
      val_test = "none",
      n_perm_tuning = 1L,
      store_payload = FALSE
    )
  })

  expect_true(all(res$results$measure_key == "mbspls.ev"))
  expect_true(all(res$results$measure_id == "mbspls.ev"))
  expect_true("Test score (mbspls.ev)" %in% res$summary_table$metric)
  expect_true("EV-weighted MAC (all LCs)" %in% res$summary_table$metric)
})


nested_cv_bootstrap_learner = function(task) {
  mbspls_graph_learner(
    learner = mlr3::lrn("clust.kmeans", centers = 2L),
    task = task,
    ncomp = 1L,
    bootstrap = TRUE,
    bootstrap_selection = TRUE,
    B = 10L,
    selection_method = "frequency",
    frequency_threshold = 0.2,
    seed_bootstrap = 1L,
    val_test = "none"
  )
}

nested_cv_env_snapshot = function(env) {
  mget(sort(ls(env, all.names = TRUE)), envir = env)
}

test_that("mbspls_nested_cv evaluates bootstrap-selection graphs with one fresh log_env per fold", {
  task = task_multiblock_synthetic(task_type = "clust", n = 60L, seed = 33L)
  gl = nested_cv_bootstrap_learner(task)
  env = gl$graph$pipeops$mbspls$param_set$values$log_env
  expect_identical(env, gl$graph$pipeops$mbspls_bootstrap_select$param_set$values$log_env)

  run = function(rs_outer) {
    mbspls_nested_cv(
      task = task, graphlearner = gl, rs_outer = rs_outer, rs_inner = mlr3::rsmp("holdout"),
      ncomp = 1L, tuner_budget = 1L, tuning_early_stop = FALSE, n_perm_tuning = 1L
    )
  }

  # A fresh learner: the selection node must find the fold's own MB-sPLS fit.
  before = nested_cv_env_snapshot(env)
  res = NULL
  expect_no_error({
    res = run(mlr3::rsmp("cv", folds = 2L))
  })
  expect_identical(nested_cv_env_snapshot(env), before)
  expect_true(all(grepl("^stable_", vapply(res$payloads, `[[`, character(1L), "weights_source"))))
  expect_true(all(is.finite(res$results$measure_test)))

  # States left by an earlier resample() must be neither used nor overwritten.
  rs_outer = mlr3::rsmp("cv", folds = 2L)
  rs_outer$instantiate(task)
  mlr3::resample(task, gl, rs_outer)
  before = nested_cv_env_snapshot(env)
  expect_true(length(before$mbspls_states) >= 2L)
  expect_no_error({
    res = run(rs_outer)
  })
  expect_identical(nested_cv_env_snapshot(env), before)
  expect_true(all(grepl("^stable_", vapply(res$payloads, `[[`, character(1L), "weights_source"))))
})

test_that("the batchtools outer-fold job evaluates bootstrap-selection graphs as configured", {
  task = task_multiblock_synthetic(task_type = "clust", n = 60L, seed = 34L)
  gl = nested_cv_bootstrap_learner(task)
  env = gl$graph$pipeops$mbspls$param_set$values$log_env
  before = nested_cv_env_snapshot(env)
  rs_outer = mlr3::rsmp("holdout")
  rs_outer$instantiate(task)
  job = NULL
  expect_no_error({
    job = .mbspls_outer_job(
      train_idx = rs_outer$train_set(1L), test_idx = rs_outer$test_set(1L), split_id = 1L,
      task = task, graphlearner = gl, rs_inner = mlr3::rsmp("holdout"), ncomp = 1L,
      tuner_budget = 1L, tuning_early_stop = FALSE, measure = mlr3::msr("mbspls.mac_evwt"),
      performance_metric = "mac", val_test = "none", val_test_n = 1L, val_test_alpha = 0.05,
      val_permute_all = TRUE, n_perm_tuning = 1L, perm_alpha_tuning = 0.05, store_payload = TRUE
    )
  })
  expect_identical(nested_cv_env_snapshot(env), before)
  expect_match(job$payload$weights_source, "^stable_")
  expect_identical(names(job), c("result_row", "c_star", "inner_score", "payload"))
  expect_true(is.finite(job$result_row$measure_test))
})

test_that("mbspls_nested_cv fits preprocessing only on the analysis rows of each outer split", {
  task = task_multiblock_synthetic(task_type = "clust", n = 45L, seed = 35L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  recorder = PipeOpRecordTrainRows$new()
  log_env = new.env(parent = emptyenv())
  gl = mlr3::as_learner(recorder %>>% mlr3pipelines::po("scale") %>>%
    PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L, append = FALSE, log_env = log_env)) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
  rs_outer = mlr3::rsmp("cv", folds = 3L)
  rs_outer$instantiate(task)
  res = mbspls_nested_cv(
    task = task, graphlearner = gl, rs_outer = rs_outer, rs_inner = mlr3::rsmp("holdout"),
    ncomp = 1L, tuner_budget = 1L, tuning_early_stop = FALSE, val_test = "permutation",
    val_test_n = 19L, n_perm_tuning = 1L
  )

  # Per split: the tuner's full-task fit, its inner training fold and the
  # evaluation fit, all restricted to the analysis rows.
  trained = recorder$record$train
  expect_length(trained, 3L * rs_outer$iters)
  for (i in seq_len(rs_outer$iters)) {
    analysis = sort(rs_outer$train_set(i))
    fits = trained[3L * (i - 1L) + 1:3]
    expect_identical(fits[[1L]], analysis)
    expect_true(all(fits[[2L]] %in% analysis) && length(fits[[2L]]) < length(analysis))
    expect_identical(fits[[3L]], analysis)
  }
  expect_identical(ls(log_env, all.names = TRUE), character(0))
  expect_true(all(res$results$val_p_last > 0 & res$results$val_p_last <= 1))
})
