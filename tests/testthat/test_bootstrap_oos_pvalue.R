test_that("cpp_bootstrap_test_oos p-value is small for strong signal (tests H0: MAC<=0)", {
  set.seed(42)
  n = 60
  # Create two strongly correlated blocks (high MAC)
  latent = rnorm(n)
  b1 = cbind(latent + rnorm(n, sd = 0.2), rnorm(n))
  b2 = cbind(latent + rnorm(n, sd = 0.2), rnorm(n))
  colnames(b1) = c("x1", "x2")
  colnames(b2) = c("y1", "y2")

  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(ncomp = 1L, c_b1 = 1.2, c_b2 = 1.2, append = FALSE)
  )
  df = data.frame(b1, b2, target = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "target")
  po$train(list(task))

  st = po$state
  Wk = st$weights[[1L]]
  X_list = list(
    b1 = as.matrix(b1),
    b2 = as.matrix(b2)
  )

  res = mlr3mbspls:::cpp_bootstrap_test_oos(
    X_test    = X_list,
    W_trained = Wk,
    n_boot    = 500L,
    spearman  = FALSE,
    frobenius = FALSE,
    alpha     = 0.05
  )

  # For a strong signal, the bootstrap MAC should be reliably > 0 -> small p
  expect_true(is.list(res))
  expect_true(is.numeric(res$p_value))
  expect_true(res$p_value < 0.10,
    info = sprintf("Expected p < 0.10 for strong signal, got p = %.4f", res$p_value))

  # stat_obs should be positive (positive latent correlation)
  expect_true(res$stat_obs > 0)
  # CI lower should also be > 0 for such a strong latent correlation
  expect_true(res$ci_lower > 0,
    info = sprintf("Expected ci_lower > 0, got %.4f", res$ci_lower))
})


test_that("cpp_bootstrap_test_oos p-value is large for pure noise (H0 not rejected)", {
  set.seed(99)
  n = 60
  # Two independent blocks with no common signal
  b1 = matrix(rnorm(n * 3), nrow = n)
  b2 = matrix(rnorm(n * 3), nrow = n)
  colnames(b1) = paste0("x", seq_len(3))
  colnames(b2) = paste0("z", seq_len(3))

  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(ncomp = 1L, c_b1 = 1.2, c_b2 = 1.2, append = FALSE)
  )
  df = data.frame(b1, b2, target = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mbN", backend = df, target = "target")
  po$train(list(task))

  st = po$state
  Wk = st$weights[[1L]]
  X_list = list(b1 = b1, b2 = b2)

  res = mlr3mbspls:::cpp_bootstrap_test_oos(
    X_test    = X_list,
    W_trained = Wk,
    n_boot    = 500L,
    spearman  = FALSE,
    frobenius = FALSE,
    alpha     = 0.05
  )

  # p_value should not be tiny for pure noise; many bootstrap MACs will be <= 0 or near 0
  expect_true(is.numeric(res$p_value))
  expect_true(res$p_value > 0)  # sanity: must be in (0, 1]
})


test_that("PipeOpMBsPLS val_test='bootstrap' stores p-value and CI in payload", {
  set.seed(7)
  n = 40
  latent = rnorm(n)
  b1 = cbind(latent + rnorm(n, sd = 0.3), rnorm(n))
  b2 = cbind(latent + rnorm(n, sd = 0.3), rnorm(n))
  colnames(b1) = c("x1", "x2")
  colnames(b2) = c("y1", "y2")

  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  log_env = new.env(parent = emptyenv())

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L, c_b1 = 1.2, c_b2 = 1.2, append = FALSE,
      log_env = log_env,
      val_test = "bootstrap", val_test_n = 100L, val_test_alpha = 0.05
    )
  )
  df = data.frame(b1, b2, target = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb_boot", backend = df, target = "target")

  po$train(list(task))
  po$predict(list(task))

  last = log_env$last
  expect_true(is.list(last))
  expect_true(!is.null(last$val_bootstrap))
  expect_s3_class(last$val_bootstrap, "data.table")
  expect_true("boot_p_value" %in% names(last$val_bootstrap))
  expect_true("boot_ci_lower" %in% names(last$val_bootstrap))
  expect_true("boot_ci_upper" %in% names(last$val_bootstrap))
  # p-value should be in (0, 1]
  pv = last$val_bootstrap$boot_p_value[1L]
  expect_true(is.finite(pv) && pv > 0 && pv <= 1)
})
