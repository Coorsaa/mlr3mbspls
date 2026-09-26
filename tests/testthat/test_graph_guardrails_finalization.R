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


test_that("graph constructors reject target columns read by site correction at prediction", {
  task = task_multiblock_synthetic(task_type = "classif", n = 30L, seed = 2L)
  target = task$target_names

  expect_error(
    mbspls_preproc_graph(
      task = task,
      site_correction = list(block_a = target),
      site_correction_methods = list(block_a = "partial_corr")
    ),
    "target column"
  )
  expect_error(
    mbsplsxy_graph(
      task = task,
      ncomp = 1L,
      site_correction = list(block_a = target),
      site_correction_methods = list(block_a = "dir")
    ),
    "target column"
  )
  expect_error(
    mbspls_preproc_graph(
      task = task,
      site_correction = list(block_a = list(site = target, covariates = character(0))),
      site_correction_methods = list(block_a = "combat")
    ),
    "target column"
  )
  # ComBat covariates are used at training time only and may name the target.
  expect_s3_class(
    mbspls_preproc_graph(
      task = task,
      site_correction = list(block_a = list(site = "site_batch", covariates = target)),
      site_correction_methods = list(block_a = "combat")
    ),
    "Graph"
  )
})


test_that("mbspls_graph rejects stable prediction weights in stability-only mode", {
  blocks = list(a = c("x1", "x2"), b = c("z1", "z2"))
  for (weights in c("stable_ci", "stable_frequency")) {
    expect_error(
      mbspls_graph(blocks = blocks, ncomp = 1L, stability_only = TRUE, predict_weights = weights),
      "stability_only = TRUE"
    )
  }
  graph = mbspls_graph(blocks = blocks, ncomp = 1L, stability_only = TRUE)
  expect_identical(graph$pipeops$mbspls$param_set$values$predict_weights, "raw")
})


test_that("graph constructors agree with PipeOpMBsPLSBootstrapSelect on alignment and seeding", {
  blocks = list(a = c("x1", "x2"), b = c("z1", "z2"))
  select_po = po("mbspls_bootstrap_select")
  levels = select_po$param_set$levels$align

  expect_setequal(eval(formals(mbspls_graph)$align), levels)
  expect_setequal(eval(formals(mbspls_graph_learner)$align), levels)

  effective_seed = function(pipeop) {
    ps = pipeop$param_set
    utils::modifyList(paradox::default_values(ps), ps$get_values(tags = "train"), keep.null = TRUE)$seed_bootstrap
  }
  graph = mbspls_graph(blocks = blocks, ncomp = 1L, align = "score_correlation")
  graph_po = graph$pipeops$mbspls_bootstrap_select
  expect_identical(graph_po$param_set$values$align, "score_correlation")
  expect_identical(effective_seed(graph_po), effective_seed(select_po))
  expect_identical(formals(mbspls_graph)$seed_bootstrap, formals(mbspls_graph_learner)$seed_bootstrap)
})
