test_that("graph constructors accept task-derived multiblock metadata", {
  task_clust = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 1L)
  site_correction = list(block_a = "site_batch", block_b = "site_batch", block_c = "site_batch")
  site_methods = list(block_a = "partial_corr", block_b = "partial_corr", block_c = "partial_corr")

  g_pre = mbspls_preproc_graph(
    task = task_clust,
    site_correction = site_correction,
    site_correction_methods = site_methods,
    k = 3L
  )
  expect_s3_class(g_pre, "Graph")

  gl_unsup = mbspls_graph_learner(
    task = task_clust,
    learner = mlr3::lrn("clust.kmeans", centers = 2L),
    site_correction = site_correction,
    site_correction_methods = site_methods,
    ncomp = 1L,
    bootstrap = FALSE,
    bootstrap_selection = FALSE,
    B = 1L,
    val_test = "none"
  )
  expect_s3_class(gl_unsup, "GraphLearner")
})


test_that("supervised graph constructor accepts TaskMultiBlock classification and regression tasks", {
  task_classif = task_multiblock_synthetic(task_type = "classif", n = 30L, seed = 1L)
  task_regr = task_multiblock_synthetic(task_type = "regr", n = 30L, seed = 1L)

  gl_classif = mbsplsxy_graph_learner(
    task = task_classif,
    ncomp = 1L
  )
  expect_s3_class(gl_classif, "GraphLearner")

  gl_regr = mbsplsxy_graph_learner(
    task = task_regr,
    ncomp = 1L
  )
  expect_s3_class(gl_regr, "GraphLearner")
})
