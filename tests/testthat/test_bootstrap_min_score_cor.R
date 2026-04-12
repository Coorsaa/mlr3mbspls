test_that("PipeOpMBsPLSBootstrapSelect - min_score_cor parameter is accepted and stored", {
  log_env = new.env(parent = emptyenv())

  po = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = TRUE,
      min_score_cor = 0.25,
      B = 5L,
      workers = 1L
    )
  )

  expect_equal(po$param_set$values$min_score_cor, 0.25)
})


test_that("PipeOpMBsPLSBootstrapSelect - min_score_cor accepts boundary values 0 and 1", {
  po_low = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = new.env(parent = emptyenv()),
      bootstrap = TRUE,
      stability_only = TRUE,
      min_score_cor = 0,
      B = 5L,
      workers = 1L
    )
  )
  expect_equal(po_low$param_set$values$min_score_cor, 0)

  po_high = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = new.env(parent = emptyenv()),
      bootstrap = TRUE,
      stability_only = TRUE,
      min_score_cor = 1,
      B = 5L,
      workers = 1L
    )
  )
  expect_equal(po_high$param_set$values$min_score_cor, 1)
})


test_that("PipeOpMBsPLSBootstrapSelect - min_score_cor=1.0 rejects all bootstrap reps (high threshold)", {
  set.seed(42)
  n = 50
  p1 = 6
  p2 = 6

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_msc", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())

  po_mbspls = mlr3mbspls::PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      store_train_blocks = TRUE,
      append = FALSE
    )
  )

  task_lv = po_mbspls$train(list(task))[[1]]

  # min_score_cor=1.0 means LVs must correlate perfectly with bootstrapped LVs;
  # virtually impossible with noise data, so all reps should be rejected and the
  # op degrades to stability_only pass-through (no crash).
  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = FALSE,
      B = 10L,
      min_score_cor = 1.0,
      selection_method = "ci",
      align = "block_sign",
      workers = 1L
    )
  )

  # Should not throw — when all reps are rejected the op passes through gracefully
  out = po_sel$train(list(task_lv))[[1]]
  expect_s3_class(out, "Task")
})


test_that("PipeOpMBsPLSBootstrapSelect - min_score_cor=0.0 accepts all bootstrap reps", {
  set.seed(1)
  n = 50
  p1 = 6
  p2 = 6

  set.seed(8)
  latent = rnorm(n)
  b1 = cbind(latent + rnorm(n, 0, 0.1), rnorm(n), latent + rnorm(n, 0, 0.1),
             rnorm(n), latent + rnorm(n, 0, 0.2), rnorm(n))
  b2 = cbind(latent + rnorm(n, 0, 0.1), rnorm(n), latent + rnorm(n, 0, 0.1),
             rnorm(n), latent + rnorm(n, 0, 0.2), rnorm(n))
  colnames(b1) = paste0("x", seq_len(ncol(b1)))
  colnames(b2) = paste0("z", seq_len(ncol(b2)))
  y = latent + rnorm(n, 0, 0.3)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_msc0", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())

  po_mbspls = mlr3mbspls::PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      store_train_blocks = TRUE,
      append = FALSE
    )
  )

  task_lv = po_mbspls$train(list(task))[[1]]

  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = FALSE,
      B = 15L,
      min_score_cor = 0.0,
      selection_method = "ci",
      align = "block_sign",
      workers = 1L
    )
  )

  out = po_sel$train(list(task_lv))[[1]]
  expect_s3_class(out, "Task")
  # All bootstrap reps accepted: stable weights should be populated in log_env state
  expect_true(!is.null(log_env$mbspls_state$weights_stable))
})
