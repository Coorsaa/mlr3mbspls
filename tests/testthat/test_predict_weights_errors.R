test_that("predict_weights='stable_ci' errors when no bootstrap-selected weights are in log_env", {
  set.seed(1)
  n = 40
  b1 = matrix(rnorm(n * 5), nrow = n, dimnames = list(NULL, paste0("x", 1:5)))
  b2 = matrix(rnorm(n * 5), nrow = n, dimnames = list(NULL, paste0("z", 1:5)))
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  log_env = new.env(parent = emptyenv())

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L, c_b1 = 2, c_b2 = 2, append = FALSE,
      log_env = log_env,
      predict_weights = "stable_ci"
    )
  )
  df = data.frame(b1, b2, y = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb_err", backend = df, target = "y")
  po$train(list(task))

  # No PipeOpMBsPLSBootstrapSelect has been run; stable_ci weights are absent
  expect_error(
    po$predict(list(task)),
    "predict_weights='stable_ci' was requested"
  )
})


test_that("predict_weights='stable_frequency' errors when no bootstrap-selected weights are in log_env", {
  set.seed(2)
  n = 40
  b1 = matrix(rnorm(n * 4), nrow = n, dimnames = list(NULL, paste0("a", 1:4)))
  b2 = matrix(rnorm(n * 4), nrow = n, dimnames = list(NULL, paste0("b", 1:4)))
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  log_env = new.env(parent = emptyenv())

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L, c_b1 = 2, c_b2 = 2, append = FALSE,
      log_env = log_env,
      predict_weights = "stable_frequency"
    )
  )
  df = data.frame(b1, b2, y = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb_err2", backend = df, target = "y")
  po$train(list(task))

  expect_error(
    po$predict(list(task)),
    "predict_weights='stable_frequency' was requested"
  )
})


test_that("predict_weights='auto' logs fallback to raw when no stable weights available", {
  set.seed(3)
  n = 40
  b1 = matrix(rnorm(n * 4), nrow = n, dimnames = list(NULL, paste0("p", 1:4)))
  b2 = matrix(rnorm(n * 4), nrow = n, dimnames = list(NULL, paste0("q", 1:4)))
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  log_env = new.env(parent = emptyenv())

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L, c_b1 = 2, c_b2 = 2, append = FALSE,
      log_env = log_env,
      predict_weights = "auto"
    )
  )
  df = data.frame(b1, b2, y = rnorm(n))
  task = mlr3::TaskRegr$new(id = "mb_auto", backend = df, target = "y")
  po$train(list(task))

  # Should succeed silently and use raw weights (no error, no stable weights present)
  out = po$predict(list(task))
  expect_true(!is.null(out[[1L]]))
  expect_s3_class(out[[1L]], "Task")
})
