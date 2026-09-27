plot_ci_state = function(w_b1) {
  list(
    blocks = list(b1 = c("a", "b", "c", "d")),
    ncomp = 1L,
    weights = list(LC_01 = list(b1 = stats::setNames(w_b1, c("a", "b", "c", "d"))))
  )
}

test_that("source = 'weights' aligns fits by the weight inner product and shows the SD", {
  testthat::skip_if_not_installed("dplyr")
  testthat::skip_if_not_installed("tibble")
  testthat::skip_if_not_installed("stringr")
  testthat::skip_if_not_installed("RColorBrewer")
  s1 = plot_ci_state(c(1, 0, 0, 0))
  # Positive inner product (0.2) although the Pearson correlation is negative.
  s2 = plot_ci_state(c(0.2, 0.98, 0, 0))

  p = mbspls_plot_block_weight_ci(list(s1, s2), source = "weights")
  means = stats::setNames(p$data$mean, as.character(p$data$feature))
  expect_equal(unname(means[c("a", "b")]), c(0.6, 0.49))
  sd_b = stats::sd(c(0, 0.98))
  row_b = p$data[as.character(p$data$feature) == "b", ]
  expect_equal(row_b$ci_high - row_b$mean, sd_b)
  expect_match(p$labels$subtitle, "mean \\+/- SD \\(descriptive")
  expect_no_match(p$labels$subtitle, "CI")

  s2_flipped = plot_ci_state(-c(0.2, 0.98, 0, 0))
  p_flipped = mbspls_plot_block_weight_ci(list(s1, s2_flipped), source = "weights")
  expect_equal(p_flipped$data$mean, p$data$mean)
})

test_that("source = 'bootstrap' labels intervals with the selector's level", {
  testthat::skip_if_not_installed("dplyr")
  testthat::skip_if_not_installed("tibble")
  testthat::skip_if_not_installed("stringr")
  testthat::skip_if_not_installed("RColorBrewer")
  fit = plot_ci_state(c(1, 0, 0, 0))
  weights_ci = data.frame(
    component = "LC_01", block = "b1", feature = c("a", "b", "c", "d"),
    boot_mean = c(0.8, 0.1, 0, 0), ci_lower = c(0.5, -0.2, 0, 0), ci_upper = c(0.9, 0.3, 0, 0)
  )
  sel = list(weights_ci = weights_ci, alpha = 0.2, alignment_method = "block_sign")
  p = mbspls_plot_block_weight_ci(
    list(mbspls = fit, mbspls_bootstrap_select = sel),
    source = "bootstrap", alpha_by_stability = FALSE
  )
  expect_match(p$labels$subtitle, "80% bootstrap percentile interval")
  expect_no_match(p$labels$subtitle, "95%")

  # Legacy draws are summarised at the same level.
  draws = data.frame(
    component = "LC_01", block = "b1", feature = "a", replicate = 1:10,
    weight = seq(0.1, 1, by = 0.1)
  )
  p_draws = mbspls_plot_block_weight_ci(
    list(mbspls = fit, mbspls_bootstrap_select = list(weights_boot_draws = draws, alpha = 0.2)),
    source = "bootstrap", alpha_by_stability = FALSE
  )
  row_a = p_draws$data[as.character(p_draws$data$feature) == "a", ]
  expect_equal(row_a$ci_low, stats::quantile(draws$weight, 0.1, names = FALSE))
  expect_equal(row_a$ci_high, stats::quantile(draws$weight, 0.9, names = FALSE))
})

test_that("excludes_zero uses the selector's magnitude threshold", {
  testthat::skip_if_not_installed("dplyr")
  testthat::skip_if_not_installed("tibble")
  testthat::skip_if_not_installed("stringr")
  testthat::skip_if_not_installed("RColorBrewer")
  fit = plot_ci_state(c(1, 0, 0, 0))
  weights_ci = data.frame(
    component = "LC_01", block = "b1", feature = c("a", "b", "c", "d"),
    boot_mean = c(0.8, 0.3, 0.002, 0), ci_lower = c(0.5, 0.1, 0.001, 0), ci_upper = c(0.9, 0.5, 0.003, 0)
  )
  plot_features = function(sel) {
    p = mbspls_plot_block_weight_ci(
      list(mbspls = fit, mbspls_bootstrap_select = sel),
      source = "bootstrap", ci_filter = "excludes_zero", alpha_by_stability = FALSE
    )
    list(features = sort(as.character(p$data$feature)), subtitle = p$labels$subtitle)
  }

  default = plot_features(list(weights_ci = weights_ci))
  expect_identical(default$features, c("a", "b", "c"))
  strict = plot_features(list(weights_ci = weights_ci, magnitude_threshold = 0.5))
  expect_identical(strict$features, "a")
  expect_match(strict$subtitle, "\\|mean\\|>0.5")
})

test_that("the bootstrap selector stores its interval level", {
  set.seed(111L)
  n = 40L
  latent = rnorm(n)
  b1 = cbind(x1 = latent + rnorm(n, 0, 0.3), x2 = rnorm(n), x3 = rnorm(n))
  b2 = cbind(z1 = latent + rnorm(n, 0, 0.3), z2 = rnorm(n), z3 = rnorm(n))
  task = mlr3::TaskRegr$new("alpha_level", data.frame(b1, b2, y = rnorm(n)), target = "y")
  log_env = new.env(parent = emptyenv())
  upstream = PipeOpMBsPLS$new(
    blocks = list(b1 = colnames(b1), b2 = colnames(b2)),
    param_vals = list(ncomp = 1L, c_b1 = 1.2, c_b2 = 1.2, log_env = log_env, store_train_blocks = TRUE)
  )
  task_lv = upstream$train(list(task))[[1L]]
  selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env, stability_only = TRUE, B = 10L, alpha = 0.2, seed_bootstrap = 112L
  ))
  suppressWarnings(selector$train(list(task_lv)))

  expect_identical(selector$state$alpha, 0.2)
  expect_identical(log_env$mbspls_state$alpha, 0.2)
  draws_level = selector$state$weights_ci
  expect_true(all(draws_level$ci_lower <= draws_level$boot_mean + 1e-12))
})

test_that("mbspls_plot_block_weight_ci reports missing suggested packages", {
  testthat::local_mocked_bindings(
    .mbspls_has_namespace = function(pkg) pkg != "dplyr",
    .package = "mlr3mbspls"
  )
  expect_error(
    mbspls_plot_block_weight_ci(list(plot_ci_state(c(1, 0, 0, 0))), source = "weights"),
    "mbspls_plot_block_weight_ci\\(\\) requires the suggested package\\(s\\) 'dplyr'"
  )
})
