flip_fixture_task = function(seed = 91L, n = 40L) {
  set.seed(seed)
  latent = rnorm(n)
  b1 = cbind(x1 = latent + rnorm(n, 0, 0.3), x2 = latent + rnorm(n, 0, 0.3), x3 = rnorm(n), x4 = rnorm(n))
  b2 = cbind(z1 = latent + rnorm(n, 0, 0.3), z2 = latent + rnorm(n, 0, 0.3), z3 = rnorm(n), z4 = rnorm(n))
  list(
    task = mlr3::TaskRegr$new("flip", data.frame(scale(b1), scale(b2), y = rnorm(n)), target = "y"),
    blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  )
}

flip_fixture_graph = function(learner = mlr3::lrn("regr.featureless")) {
  fx = flip_fixture_task()
  log_env = new.env(parent = emptyenv())
  graph = mlr3pipelines::po("mbspls",
    blocks = fx$blocks, ncomp = 2L, c_b1 = sqrt(2), c_b2 = sqrt(2),
    log_env = log_env, store_train_blocks = TRUE, append = TRUE,
    predict_weights = "raw"
  ) %>>%
    mlr3pipelines::po("mbspls_bootstrap_select",
      log_env = log_env, B = 10L, seed_bootstrap = 92L, selection_method = "frequency"
    ) %>>%
    mlr3pipelines::po("learner", learner)
  gl = mlr3::as_learner(graph)
  list(gl = gl, task = fx$task, log_env = log_env)
}

test_that("the list method flips weights, loadings and scores and keeps names", {
  fx = flip_fixture_task()
  po_fit = PipeOpMBsPLS$new(blocks = fx$blocks, param_vals = list(
    ncomp = 2L, c_b1 = sqrt(2), c_b2 = sqrt(2)
  ))
  po_fit$train(list(fx$task))
  st = po_fit$state
  st$weights_ci = data.frame(
    component = "LC_01", block = c("b1", "b2"), feature = c("x1", "z1"),
    boot_mean = c(0.5, 0.6), ci_lower = c(0.2, 0.3), ci_upper = c(0.8, NA)
  )

  flipped = mbspls_flip_weights(st)
  expect_equal(flipped$weights$LC_01$b1, -st$weights$LC_01$b1)
  expect_identical(names(flipped$weights$LC_02$b2), names(st$weights$LC_02$b2))
  expect_identical(dim(flipped$weights$LC_01$b1), dim(st$weights$LC_01$b1))
  expect_equal(flipped$loadings$LC_02$b1, -st$loadings$LC_02$b1)
  expect_equal(flipped$T_mat, -st$T_mat)
  expect_equal(flipped$weights_ci$ci_lower, c(-0.8, NA))
  expect_equal(flipped$weights_ci$ci_upper, c(-0.2, -0.3))
  expect_equal(mbspls_flip_weights(flipped), st)

  signs = matrix(c(1, 1, -1, 1), 2L, dimnames = list(c("LC_01", "LC_02"), c("b1", "b2")))
  partial = mbspls_flip_weights(st, signs = signs)
  expect_equal(partial$weights$LC_01$b1, st$weights$LC_01$b1)
  expect_equal(partial$weights$LC_01$b2, -st$weights$LC_01$b2)
  expect_equal(partial$T_mat[, "LV1_b2"], -st$T_mat[, "LV1_b2"])
  expect_equal(partial$T_mat[, "LV2_b2"], st$T_mat[, "LV2_b2"])

  expect_error(mbspls_flip_weights(st, signs = 0.5), "must be -1 or 1")
})

test_that("the PipeOp method flips its state and the log_env run entry", {
  fx = flip_fixture_task()
  log_env = new.env(parent = emptyenv())
  po_fit = PipeOpMBsPLS$new(blocks = fx$blocks, param_vals = list(
    ncomp = 2L, c_b1 = sqrt(2), c_b2 = sqrt(2), log_env = log_env, predict_weights = "raw"
  ))
  po_fit$train(list(fx$task))
  before = po_fit$state
  pred_before = po_fit$predict(list(fx$task))[[1L]]$data(cols = colnames(before$T_mat))
  payload_before = log_env$mbspls_last[[before$run_id]]
  expect_false(is.null(payload_before$T_mat))

  copy = mbspls_flip_weights(po_fit, inplace = FALSE)
  expect_equal(copy$weights$LC_01$b1, -before$weights$LC_01$b1)
  expect_identical(po_fit$state, before)
  expect_identical(log_env$mbspls_last[[before$run_id]], payload_before)

  mbspls_flip_weights(po_fit)
  expect_equal(po_fit$state$weights$LC_02$b2, -before$weights$LC_02$b2)
  # the stored prediction payload is flipped with the fit; sign-free
  # statistics are unchanged
  payload = log_env$mbspls_last[[before$run_id]]
  expect_equal(payload$T_mat, -payload_before$T_mat)
  expect_equal(payload$mac_comp, payload_before$mac_comp)
  expect_equal(log_env$last$T_mat, -payload_before$T_mat)
  pred_after = po_fit$predict(list(fx$task))[[1L]]$data(cols = colnames(before$T_mat))
  expect_equal(as.matrix(pred_after), -as.matrix(pred_before))

  entry = log_env$mbspls_states[[before$run_id]]
  expect_equal(entry$weights$LC_01$b1, -before$weights$LC_01$b1)
  expect_equal(entry$T_mat_train, -before$T_mat)
  expect_equal(log_env$mbspls_state$weights, entry$weights)
})

test_that("the GraphLearner method keeps fit, bootstrap selection and log_env in sync", {
  fx = flip_fixture_graph()
  gl = fx$gl
  expect_error(mbspls_flip_weights(gl), "not trained")
  gl$train(fx$task)
  gl$predict(fx$task)
  fit_before = gl$model$mbspls
  sel_before = gl$model$mbspls_bootstrap_select
  env_before = fx$log_env$mbspls_states[[fit_before$run_id]]
  last_before = fx$log_env$last$T_mat

  model_copy = mbspls_flip_weights(gl, inplace = FALSE)
  expect_equal(model_copy$mbspls$weights$LC_01$b1, -fit_before$weights$LC_01$b1)
  expect_identical(gl$model$mbspls, fit_before)

  expect_no_warning(mbspls_flip_weights(gl))
  expect_equal(gl$model$mbspls$weights$LC_01$b1, -fit_before$weights$LC_01$b1)
  expect_equal(fx$log_env$last$T_mat, -last_before)
  expect_equal(fx$log_env$mbspls_last[[fit_before$run_id]]$T_mat, -last_before)
  sel = gl$model$mbspls_bootstrap_select
  expect_equal(sel$weights_stable$LC_01$b2, -sel_before$weights_stable$LC_01$b2)
  expect_equal(sel$loadings_stable$LC_02$b1, -sel_before$loadings_stable$LC_02$b1)
  expect_equal(sel$weights_ci$boot_mean, -sel_before$weights_ci$boot_mean)
  expect_true(all(sel$weights_ci$ci_lower <= sel$weights_ci$ci_upper, na.rm = TRUE))
  expect_equal(sel$weights_selectfreq, sel_before$weights_selectfreq)
  env_after = fx$log_env$mbspls_states[[fit_before$run_id]]
  expect_equal(env_after$weights_stable$LC_01$b1, -env_before$weights_stable$LC_01$b1)
  expect_equal(env_after$weights$LC_02$b2, -env_before$weights$LC_02$b2)

  gl$predict(fx$task)
  expect_equal(fx$log_env$last$T_mat, -last_before)

  mbspls_flip_weights(gl)
  expect_equal(gl$model$mbspls, fit_before)
  expect_equal(gl$model$mbspls_bootstrap_select, sel_before)
  expect_equal(fx$log_env$mbspls_states[[fit_before$run_id]], env_before)
  expect_equal(fx$log_env$last$T_mat, last_before)
})

test_that("in-place flips are refused when a downstream model used the scores", {
  fx = flip_fixture_graph(learner = mlr3::lrn("regr.rpart"))
  fx$gl$train(fx$task)
  fit_before = fx$gl$model$mbspls
  expect_error(mbspls_flip_weights(fx$gl), "Cannot flip signs in place.*regr.rpart")
  expect_identical(fx$gl$model$mbspls, fit_before)
  expect_equal(
    mbspls_flip_weights(fx$gl, inplace = FALSE)$mbspls$weights$LC_01$b1,
    -fit_before$weights$LC_01$b1
  )
})
