extract_fixture = function(alpha = 0.1) {
  set.seed(101L)
  n = 50L
  latent = rnorm(n)
  b1 = cbind(x1 = latent + rnorm(n, 0, 0.3), x2 = latent + rnorm(n, 0, 0.4), x3 = rnorm(n), x4 = rnorm(n))
  b2 = cbind(z1 = latent + rnorm(n, 0, 0.3), z2 = latent + rnorm(n, 0, 0.4), z3 = rnorm(n))
  task = mlr3::TaskRegr$new("extract", data.frame(scale(b1), scale(b2), y = rnorm(n)), target = "y")
  log_env = new.env(parent = emptyenv())
  graph = mlr3pipelines::po("mbspls",
    blocks = list(b1 = colnames(b1), b2 = colnames(b2)), ncomp = 1L,
    c_b1 = 1.5, c_b2 = 1.5, log_env = log_env, store_train_blocks = TRUE
  ) %>>%
    mlr3pipelines::po("mbspls_bootstrap_select",
      log_env = log_env, stability_only = TRUE, B = 10L, alpha = alpha,
      seed_bootstrap = 102L, frequency_threshold = 0.7
    ) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))
  gl = mlr3::as_learner(graph)
  suppressWarnings(gl$train(task))
  gl
}

test_that("bootstrap means are read from the stored aligned summaries", {
  gl = extract_fixture()
  st = gl$model$mbspls_bootstrap_select
  ci = data.table::as.data.table(st$weights_ci)

  out = mbspls_extract_bootstrap_means(st)
  expected = ci[(ci$ci_lower > 0 | ci$ci_upper < 0) & abs(ci$boot_mean) > 1e-3]
  expect_setequal(paste(out$block, out$feature), paste(expected$block, expected$feature))
  expect_equal(out$mean, expected$boot_mean[match(paste(out$block, out$feature), paste(expected$block, expected$feature))])
  expect_identical(names(out), c("component", "block", "feature", "mean", "ci_low", "ci_high", "freq"))
  expect_identical(out, mbspls_extract_bootstrap_means(gl))
  expect_identical(out, mbspls_extract_bootstrap_means(list(mbspls_bootstrap_select = st)))

  freq = mbspls_extract_bootstrap_means(st, filter_method = "frequency")
  fr = st$weights_selectfreq
  expect_setequal(freq$feature, fr$feature[fr$freq >= 0.7])
  expect_setequal(
    mbspls_extract_bootstrap_means(st, filter_method = "frequency", filter_level = 0)$feature,
    fr$feature
  )

  expect_error(
    mbspls_extract_bootstrap_means(st, filter_level = 0.95),
    "Retrain the bootstrap selection with alpha = 0.05"
  )
  expect_identical(mbspls_extract_bootstrap_means(st, filter_level = 0.9), out)

  no_freq = st
  no_freq$weights_selectfreq = NULL
  plain = mbspls_extract_bootstrap_means(no_freq)
  expect_true(all(is.na(plain$freq)))
  expect_error(mbspls_extract_bootstrap_means(no_freq, filter_method = "frequency"), "weights_selectfreq")
  expect_error(mbspls_extract_bootstrap_means(gl$model$mbspls), "Could not find")
})

test_that("filter_level is ignored with a warning when the interval level is unknown", {
  gl = extract_fixture()
  st = gl$model$mbspls_bootstrap_select
  legacy = st
  legacy$alpha = NULL
  expect_warning(
    {
      out = mbspls_extract_bootstrap_means(legacy, filter_level = 0.5)
    },
    "does not record the level of its stored intervals"
  )
  expect_identical(out, mbspls_extract_bootstrap_means(st))
  expect_no_warning(mbspls_extract_bootstrap_means(legacy))
})

test_that("legacy draws are summarised as already aligned draws", {
  draws = data.frame(
    component = "LC_01", block = "b",
    feature = rep(c("f1", "f2"), each = 10L), replicate = rep(1:10, 2L),
    weight = c(seq(0.1, 1, by = 0.1), c(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1))
  )
  st = list(weights_boot_draws = draws, alpha = 0.2)
  out = mbspls_extract_bootstrap_means(st, filter_method = "ci")
  expect_identical(out$feature, "f1")
  expect_equal(out$ci_low, stats::quantile(seq(0.1, 1, by = 0.1), 0.1, names = FALSE))
  expect_equal(out$mean, 0.55)
})

test_that("mbspls_extract_bootstrap_means needs no suggested package", {
  gl = extract_fixture()
  testthat::local_mocked_bindings(
    .mbspls_has_namespace = function(pkg) FALSE,
    .package = "mlr3mbspls"
  )
  expect_s3_class(mbspls_extract_bootstrap_means(gl), "data.table")
})
