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

test_that("mbspls_eval_new_data works with custom MB-sPLS node id", {
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(203)

  n = 40
  X1 = matrix(rnorm(n * 4), nrow = n, ncol = 4)
  colnames(X1) = paste0("x1_", seq_len(ncol(X1)))
  X2 = matrix(rnorm(n * 6), nrow = n, ncol = 6)
  colnames(X2) = paste0("x2_", seq_len(ncol(X2)))
  y = rnorm(n)

  df = data.frame(X1, X2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_eval_custom_id", backend = df, target = "y")
  blocks = list(
    b1 = colnames(X1),
    b2 = colnames(X2)
  )

  graph = po(
    "mbspls",
    id = "mbspls_custom",
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

  res = mbspls_eval_new_data(gl = gl, task = task)

  expect_type(res, "list")
  expect_true(is.numeric(res$mac_comp))
  expect_length(res$mac_comp, 2L)
  expect_true(is.matrix(res$ev_block))
  expect_equal(dim(res$ev_block), c(2L, 2L))
})

test_that("mbspls_plot_block_weight_ci works with custom MB-sPLS node id", {
  testthat::skip_if_not_installed("ggplot2")
  testthat::skip_if_not_installed("dplyr")
  testthat::skip_if_not_installed("tibble")
  testthat::skip_if_not_installed("stringr")
  testthat::skip_if_not_installed("RColorBrewer")
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(204)

  n = 36
  X1 = matrix(rnorm(n * 4), nrow = n, ncol = 4)
  colnames(X1) = paste0("x1_", seq_len(ncol(X1)))
  X2 = matrix(rnorm(n * 5), nrow = n, ncol = 5)
  colnames(X2) = paste0("x2_", seq_len(ncol(X2)))
  y = rnorm(n)

  df = data.frame(X1, X2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_plot_custom_id", backend = df, target = "y")
  blocks = list(
    b1 = colnames(X1),
    b2 = colnames(X2)
  )

  graph = po(
    "mbspls",
    id = "mbspls_custom",
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

  p = mbspls_plot_block_weight_ci(gl, source = "weights")

  expect_s3_class(p, "ggplot")
})

test_that("mbspls_plot_block_weight_ci bootstrap path works with custom node ids", {
  testthat::skip_if_not_installed("ggplot2")
  testthat::skip_if_not_installed("dplyr")
  testthat::skip_if_not_installed("tibble")
  testthat::skip_if_not_installed("stringr")
  testthat::skip_if_not_installed("RColorBrewer")
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(205)

  n = 32
  X1 = matrix(rnorm(n * 4), nrow = n, ncol = 4)
  colnames(X1) = paste0("x1_", seq_len(ncol(X1)))
  X2 = matrix(rnorm(n * 5), nrow = n, ncol = 5)
  colnames(X2) = paste0("x2_", seq_len(ncol(X2)))
  y = rnorm(n)

  df = data.frame(X1, X2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_plot_bootstrap_custom_ids", backend = df, target = "y")
  log_env = new.env(parent = emptyenv())
  blocks = list(
    b1 = colnames(X1),
    b2 = colnames(X2)
  )

  graph = po(
    "mbspls",
    id = "mbspls_custom",
    blocks = blocks,
    ncomp = 2L,
    c_b1 = 2L,
    c_b2 = 2L,
    permutation_test = FALSE,
    val_test = "none",
    store_train_blocks = TRUE,
    append = TRUE,
    log_env = log_env
  ) %>>%
    po(
      "mbspls_bootstrap_select",
      id = "mbspls_bootstrap_select_custom",
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = FALSE,
      B = 5L,
      selection_method = "ci",
      align = "block_sign",
      workers = 1L
    ) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))

  gl = mlr3::as_learner(graph)
  gl$train(task)

  p = mbspls_plot_block_weight_ci(gl, source = "bootstrap")

  expect_s3_class(p, "ggplot")
  expect_identical(levels(p$data$component_lab), c("LC 1", "LC 2"))
})


test_that("mbspls_eval_new_data restores the original log_env on template and fitted nodes", {
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(206)
  n = 30
  X1 = matrix(rnorm(n * 3), nrow = n, ncol = 3)
  colnames(X1) = paste0("x1_", seq_len(ncol(X1)))
  X2 = matrix(rnorm(n * 4), nrow = n, ncol = 4)
  colnames(X2) = paste0("x2_", seq_len(ncol(X2)))
  y = rnorm(n)

  df = data.frame(X1, X2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_eval_restore_env", backend = df, target = "y")
  blocks = list(b1 = colnames(X1), b2 = colnames(X2))
  orig_env = new.env(parent = emptyenv())

  graph = po(
    "mbspls",
    blocks = blocks,
    ncomp = 2L,
    c_b1 = 1L,
    c_b2 = 2L,
    permutation_test = FALSE,
    val_test = "none",
    log_env = orig_env
  ) %>>% mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))

  gl = mlr3::as_learner(graph)
  gl$train(task)

  expect_identical(gl$graph$pipeops$mbspls$param_set$values$log_env, orig_env)
  fitted_env_before = tryCatch(gl$model$mbspls$param_set$values$log_env, error = function(e) NULL)

  invisible(mbspls_eval_new_data(gl = gl, task = task))

  expect_identical(gl$graph$pipeops$mbspls$param_set$values$log_env, orig_env)
  fitted_env_after = tryCatch(gl$model$mbspls$param_set$values$log_env, error = function(e) NULL)
  if (inherits(fitted_env_before, "environment")) {
    expect_identical(fitted_env_after, fitted_env_before)
  }
})


test_that("mbspls_eval_new_data evaluates the stability-selected weights the pipeline uses", {
  testthat::skip_if_not("regr.featureless" %in% mlr3::mlr_learners$keys())

  set.seed(207)
  n = 90L
  l1 = rnorm(n)
  l2 = rnorm(n)
  make_block = function(prefix) {
    m = cbind(l1 + rnorm(n, sd = 0.4), l2 + rnorm(n, sd = 0.6), matrix(rnorm(n * 4L), n))
    colnames(m) = paste0(prefix, seq_len(ncol(m)))
    m
  }
  X1 = make_block("x")
  X2 = make_block("z")
  task = mlr3::TaskRegr$new("mb_eval_stable", data.frame(X1, X2, y = l1 + rnorm(n)), target = "y")
  blocks = list(b1 = colnames(X1), b2 = colnames(X2))

  for (predict_weights in c("auto", "stable_ci")) {
    log_env = new.env(parent = emptyenv())
    log_env$warn_overwrite = FALSE
    gl = mbspls_graph_learner(
      learner = mlr3::lrn("regr.featureless"),
      blocks = blocks,
      ncomp = 2L,
      B = 30L,
      predict_weights = predict_weights,
      log_env = log_env
    )
    gl$train(task)
    gl$predict(task)
    reference = log_env$last
    last_before = log_env$last
    states_before = log_env$mbspls_states

    res = mbspls_eval_new_data(gl, task)

    expect_identical(res$weights_source, reference$weights_source)
    expect_match(res$weights_source, "^stable_")
    expect_equal(res$mac_comp, reference$mac_comp)
    expect_equal(res$ev_block, reference$ev_block)
    expect_equal(res$ev_comp, reference$ev_comp)
    expect_equal(res$T_mat, reference$T_mat)
    stable = log_env$mbspls_states[[gl$model$mbspls$run_id]]$weights_stable
    expect_equal(res$weights, stable, ignore_attr = TRUE)
    expect_identical(res$weights_raw, gl$model$mbspls$weights)
    expect_identical(res$ncomp, 2L)
    expect_identical(log_env$last, last_before)
    expect_identical(log_env$mbspls_states, states_before)
  }
})
