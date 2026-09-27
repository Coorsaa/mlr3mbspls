test_that("cpp_bootstrap_test_oos reports descriptive uncertainty for signal", {
  set.seed(42)
  n = 60
  latent = rnorm(n)
  block1 = cbind(latent + rnorm(n, sd = 0.2), rnorm(n))
  block2 = cbind(latent + rnorm(n, sd = 0.2), rnorm(n))
  colnames(block1) = c("x1", "x2")
  colnames(block2) = c("y1", "y2")

  blocks = list(b1 = colnames(block1), b2 = colnames(block2))
  pipeop = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(ncomp = 1L, c_b1 = 1.2, c_b2 = 1.2,
      append = FALSE)
  )
  data = data.frame(block1, block2, target = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb", backend = data, target = "target")
  pipeop$train(list(task))

  result = mlr3mbspls:::cpp_bootstrap_test_oos(
    X_test = list(b1 = as.matrix(block1), b2 = as.matrix(block2)),
    W_trained = pipeop$state$weights[[1L]],
    n_boot = 500L,
    spearman = FALSE,
    frobenius = FALSE,
    alpha = 0.05
  )

  expect_true(is.list(result))
  expect_true(is.na(result$p_value))
  expect_match(result$p_value_note, "not a null distribution",
    ignore.case = TRUE)
  expect_equal(result$bias, result$boot_mean - result$stat_obs)
  expect_identical(result$interval_type, "percentile")
  expect_identical(result$n_boot, 500L)
  expect_length(result$replicates, 500L)
  expect_gt(result$stat_obs, 0)
  expect_gt(result$ci_lower, 0)
})

test_that("cpp_bootstrap_test_oos never turns noise bootstrap into a p-value", {
  set.seed(99)
  n = 60
  block1 = matrix(rnorm(n * 3), nrow = n)
  block2 = matrix(rnorm(n * 3), nrow = n)
  colnames(block1) = paste0("x", seq_len(3))
  colnames(block2) = paste0("z", seq_len(3))

  blocks = list(b1 = colnames(block1), b2 = colnames(block2))
  pipeop = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(ncomp = 1L, c_b1 = 1.2, c_b2 = 1.2,
      append = FALSE)
  )
  data = data.frame(block1, block2, target = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb_noise", backend = data,
    target = "target")
  pipeop$train(list(task))

  result = mlr3mbspls:::cpp_bootstrap_test_oos(
    X_test = list(b1 = block1, b2 = block2),
    W_trained = pipeop$state$weights[[1L]],
    n_boot = 200L,
    spearman = FALSE,
    frobenius = FALSE,
    alpha = 0.05
  )

  expect_true(is.na(result$p_value))
  expect_equal(result$n_boot_requested,
    result$n_boot + result$n_boot_failed)
  expect_lte(result$ci_lower, result$ci_upper)
})

test_that("PipeOpMBsPLS bootstrap payload uses the descriptive schema", {
  set.seed(7)
  n = 40
  latent = rnorm(n)
  block1 = cbind(latent + rnorm(n, sd = 0.3), rnorm(n))
  block2 = cbind(latent + rnorm(n, sd = 0.3), rnorm(n))
  colnames(block1) = c("x1", "x2")
  colnames(block2) = c("y1", "y2")

  blocks = list(b1 = colnames(block1), b2 = colnames(block2))
  log_env = new.env(parent = emptyenv())
  pipeop = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 1.2,
      c_b2 = 1.2,
      append = FALSE,
      log_env = log_env,
      val_test = "bootstrap",
      val_test_n = 100L,
      val_test_alpha = 0.05,
      seed_validation = 808L
    )
  )
  data = data.frame(block1, block2, target = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb_boot", backend = data,
    target = "target")

  pipeop$train(list(task))
  rng_before = .Random.seed
  pipeop$predict(list(task))
  first_vectors = log_env$last$val_boot_vectors
  expect_identical(.Random.seed, rng_before)
  pipeop$predict(list(task))
  expect_identical(log_env$last$val_boot_vectors, first_vectors)
  expect_identical(.Random.seed, rng_before)

  payload = log_env$last
  expect_true(is.list(payload))
  expect_s3_class(payload$val_bootstrap, "data.table")
  expect_true(all(c(
    "estimate", "bootstrap_mean", "bias", "standard_error",
    "conf_low", "conf_high", "interval_type", "replicates_effective",
    "p_value", "p_value_note"
  ) %in% names(payload$val_bootstrap)))
  expect_true(is.na(payload$val_bootstrap$p_value[[1L]]))
  expect_true(is.na(payload$val_bootstrap$boot_p_value[[1L]]))
  expect_false("val_test_p" %in% names(payload))
  expect_length(payload$val_boot_vectors[[1L]], 100L)
  expect_identical(payload$val_test_params$seed, 808L)
  expect_length(payload$val_test_params$rng_streams, 1L)
  expect_match(payload$val_test_params$scope, "no null-hypothesis p-value")
})

test_that("out-of-sample native validation rejects schema and row mismatches", {
  x = list(matrix(rnorm(20), 10, 2), matrix(rnorm(24), 12, 2))
  weights = list(c(1, 0), c(1, 0))

  expect_error(
    mlr3mbspls:::cpp_perm_test_oos(x, weights, n_perm = 9L),
    "row mismatch"
  )
  expect_error(
    mlr3mbspls:::cpp_bootstrap_test_oos(x, weights, n_boot = 10L),
    "row mismatch"
  )
  expect_error(
    mlr3mbspls:::cpp_perm_test_oos(x[1L], weights, n_perm = 9L),
    "at least 2 blocks|one trained weight"
  )
})
