test_that("supervised graph constructors fail fast on incompatible task or learner types", {
  task_clust = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 1L)
  task_classif = task_multiblock_synthetic(task_type = "classif", n = 30L, seed = 1L)

  expect_error(
    mbsplsxy_graph(task = task_clust, ncomp = 1L),
    "requires a classification or regression task"
  )

  expect_error(
    mbsplsxy_graph_learner(
      task = task_classif,
      learner = mlr3::lrn("regr.featureless"),
      ncomp = 1L
    ),
    "does not match the expected task type"
  )
})


test_that("preprocessing graph validates site-correction columns when a task is supplied", {
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 1L)

  expect_error(
    mbspls_preproc_graph(
      task = task,
      site_correction = list(block_a = "missing_site_column"),
      site_correction_methods = list(block_a = "partial_corr")
    ),
    "site-correction columns not found"
  )
})
