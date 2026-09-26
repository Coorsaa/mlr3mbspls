two_factor_graph_learner = function(log_env, selection_method = "frequency") {
  set.seed(121L)
  n = 60L
  f1 = rnorm(n)
  f2 = rnorm(n)
  b1 = cbind(
    x1 = f1 + rnorm(n, 0, 0.3), x2 = f1 + rnorm(n, 0, 0.3),
    x3 = f2 + rnorm(n, 0, 0.3), x4 = f2 + rnorm(n, 0, 0.3), x5 = rnorm(n)
  )
  b2 = cbind(
    z1 = f1 + rnorm(n, 0, 0.3), z2 = f1 + rnorm(n, 0, 0.3),
    z3 = f2 + rnorm(n, 0, 0.3), z4 = f2 + rnorm(n, 0, 0.3), z5 = rnorm(n)
  )
  task = mlr3::TaskRegr$new("two_factor", data.frame(scale(b1), scale(b2), y = rnorm(n)), target = "y")
  graph = mlr3pipelines::po("mbspls",
    blocks = list(b1 = colnames(b1), b2 = colnames(b2)), ncomp = 2L,
    c_b1 = sqrt(2), c_b2 = sqrt(2), log_env = log_env, store_train_blocks = TRUE,
    append = TRUE
  ) %>>%
    mlr3pipelines::po("mbspls_bootstrap_select",
      log_env = log_env, B = 20L, seed_bootstrap = 122L, selection_method = selection_method
    ) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))
  list(gl = mlr3::as_learner(graph), task = task)
}

test_that("freq_min filters the bootstrap means for every component", {
  log_env = new.env(parent = emptyenv())
  fx = two_factor_graph_learner(log_env)
  fx$gl$train(fx$task)
  fit = fx$gl$model$mbspls
  sel = fx$gl$model$mbspls_bootstrap_select
  weights_from_source = getFromNamespace(".mbspls_weights_from_source", "mlr3mbspls")
  recompute = getFromNamespace(".mbspls_recompute_from_weights", "mlr3mbspls")

  W = weights_from_source(fit, sel, "bootstrap", freq_min = 0.8)
  expect_identical(names(W), c("LC_01", "LC_02"))
  for (k in names(W)) {
    expect_identical(names(W[[k]]), c("b1", "b2"))
  }
  ci = sel$weights_ci
  fr = sel$weights_selectfreq
  for (k in names(W)) {
    for (b in names(W[[k]])) {
      w = W[[k]][[b]]
      freq = fr$freq[fr$component == k & fr$block == b][match(names(w), fr$feature[fr$component == k & fr$block == b])]
      mu = ci$boot_mean[ci$component == k & ci$block == b][match(names(w), ci$feature[ci$component == k & ci$block == b])]
      expect_equal(unname(w), ifelse(freq >= 0.8, mu, 0))
    }
  }
  expect_true(any(W$LC_02$b1 != 0))

  rec = recompute(fit, W, log_env = log_env)
  expect_true(all(rec$ev_comp > 0))

  testthat::skip_if_not_installed("scales")
  p = ggplot2::autoplot(fx$gl, type = "mbspls_variance", source = "bootstrap", freq_min = 0.8)
  expect_s3_class(p, "ggplot")
})

test_that("recomputation rejects weights that are not named by block", {
  log_env = new.env(parent = emptyenv())
  fx = two_factor_graph_learner(log_env)
  fx$gl$train(fx$task)
  fit = fx$gl$model$mbspls
  recompute = getFromNamespace(".mbspls_recompute_from_weights", "mlr3mbspls")
  W = fit$weights
  names(W$LC_02) = NULL
  expect_error(recompute(fit, W, log_env = log_env), "named by block")
})

test_that("the evaluation payload belongs to the learner's own training run", {
  log_env = new.env(parent = emptyenv())
  # two runs share the environment on purpose
  log_env$warn_overwrite = FALSE
  first = two_factor_graph_learner(log_env)
  second = two_factor_graph_learner(log_env)
  first$gl$train(first$task)
  get_payload = getFromNamespace(".mbspls_get_eval_payload", "mlr3mbspls")
  expect_error(get_payload(first$gl), "No evaluation payload found for the fitted run")

  first$gl$predict(first$task)
  second$gl$param_set$values$mbspls.c_b1 = 1
  second$gl$train(second$task)
  second$gl$predict(second$task)

  first_run = first$gl$model$mbspls$run_id
  expect_false(identical(first_run, second$gl$model$mbspls$run_id))
  expect_identical(get_payload(first$gl), log_env$mbspls_last[[first_run]])
  expect_false(identical(get_payload(first$gl), log_env$last))
  expect_identical(get_payload(second$gl), log_env$last)
})

test_that("plot helpers report missing suggested packages", {
  log_env = new.env(parent = emptyenv())
  fx = two_factor_graph_learner(log_env)
  fx$gl$train(fx$task)
  testthat::local_mocked_bindings(
    .mbspls_has_namespace = function(pkg) !pkg %in% c("ggraph", "patchwork"),
    .package = "mlr3mbspls"
  )
  expect_error(
    ggplot2::autoplot(fx$gl, type = "mbspls_network"),
    "requires the suggested package\\(s\\) 'ggraph'"
  )
  expect_error(
    ggplot2::autoplot(fx$gl, type = "mbspls_weights"),
    "requires the suggested package\\(s\\) 'patchwork'"
  )
})
