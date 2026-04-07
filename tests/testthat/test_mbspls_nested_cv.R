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

  expect_no_error(
    mbspls_nested_cv(
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

  expect_null(gl$model)
  expect_null(gl$graph$pipeops[[mbspls_id]]$param_set$values$c_matrix)
})
