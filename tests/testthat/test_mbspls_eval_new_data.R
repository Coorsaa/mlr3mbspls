test_that("mbspls_eval_new_data evaluates a trained GraphLearner on new blocks", {
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(202)

  n = 50
  X1 = matrix(rnorm(n * 5), nrow = n, ncol = 5)
  colnames(X1) = paste0("x1_", seq_len(ncol(X1)))
  X2 = matrix(rnorm(n * 7), nrow = n, ncol = 7)
  colnames(X2) = paste0("x2_", seq_len(ncol(X2)))
  y = rnorm(n)

  df = data.frame(X1, X2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_eval", backend = df, target = "y")
  blocks = list(
    b1 = colnames(X1),
    b2 = colnames(X2)
  )

  graph = po(
    "mbspls",
    blocks = blocks,
    ncomp = 2L,
    c_b1 = 2L,
    c_b2 = 2L,
    permutation_test = FALSE,
    val_test = "none"
  ) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))

  gl = mlr3::as_learner(graph)
  gl$train(task)

  # Evaluate on the same data but through the public helper.
  res = mbspls_eval_new_data(gl = gl, task = task)

  expect_type(res, "list")
  expect_true(is.list(res$weights))
  expect_true(is.list(res$loadings))
  expect_true(is.list(res$blocks_map))
  expect_true(is.character(res$weights_source))

  expect_true(is.numeric(res$mac_comp))
  expect_length(res$mac_comp, 2L)

  # Explained-variance payloads are present for regr tasks.
  expect_true(is.numeric(res$ev_comp))
  expect_length(res$ev_comp, 2L)
  expect_true(is.matrix(res$ev_block))
  expect_equal(dim(res$ev_block), c(2L, 2L))
})

test_that("mbspls_eval_new_data errors if the graph is untrained", {
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(1)
  df = data.frame(x1 = rnorm(20), x2 = rnorm(20), y = rnorm(20))
  task = mlr3::TaskRegr$new(id = "t", backend = df, target = "y")

  graph = mlr3pipelines::po("scale") %>>%
    mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))
  gl = mlr3::as_learner(graph)

  expect_error(
    mbspls_eval_new_data(gl = gl, task = task),
    regexp = "untrained",
    ignore.case = TRUE
  )
})

test_that("mbspls_eval_new_data errors if the graph has no PipeOpMBsPLS", {
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(1)
  df = data.frame(x1 = rnorm(20), x2 = rnorm(20), y = rnorm(20))
  task = mlr3::TaskRegr$new(id = "t2", backend = df, target = "y")

  graph = mlr3pipelines::po("scale") %>>%
    mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))
  gl = mlr3::as_learner(graph)
  gl$train(task)

  expect_error(
    mbspls_eval_new_data(gl = gl, task = task),
    regexp = "PipeOpMBsPLS|mbspls",
    ignore.case = TRUE
  )
})
