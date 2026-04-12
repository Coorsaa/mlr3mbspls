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
  expect_no_error(
    res <- mbspls_nested_cv(
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
  )

  expect_true(all(c("measure_id", "measure_key", "measure_test", "measure_test_defined", "measure_test_status") %in% names(res$results)))
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
  expect_no_error(
    res <- mbspls_nested_cv(
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
  )

  expect_true(all(res$results$measure_key == "mbspls.ev"))
  expect_true(all(res$results$measure_id == "mbspls.ev"))
  expect_true("Test score (mbspls.ev)" %in% res$summary_table$metric)
  expect_true("EV-weighted MAC (all LCs)" %in% res$summary_table$metric)
})
