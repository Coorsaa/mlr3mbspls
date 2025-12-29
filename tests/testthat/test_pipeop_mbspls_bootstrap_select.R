test_that("PipeOpMBsPLSBootstrapSelect - stability_only TRUE is a pass-through", {
  set.seed(1)

  n = 40
  p1 = 6
  p2 = 8

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
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
  dt_before = as.data.frame(task_lv$data())
  feats_before = task_lv$feature_names

  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = TRUE,
      B = 5L,
      selection_method = "ci",
      align = "block_sign",
      workers = 1L
    )
  )

  out_train = po_sel$train(list(task_lv))[[1]]

  # Pass-through means: no feature dropping and no LV replacement.
  expect_setequal(out_train$feature_names, feats_before)
  expect_equal(as.data.frame(out_train$data()), dt_before)

  # Predict should also pass-through unchanged in stability_only mode.
  task_lv_new = task_lv$clone(deep = TRUE)
  task_lv_new$filter(1:10)
  out_pred = po_sel$predict(list(task_lv_new))[[1]]
  expect_setequal(out_pred$feature_names, task_lv_new$feature_names)
  expect_equal(as.data.frame(out_pred$data()), as.data.frame(task_lv_new$data()))
})


test_that("PipeOpMBsPLSBootstrapSelect - errors if blocks cannot be rebuilt and X_train_blocks is missing", {
  set.seed(2)

  n = 30
  p1 = 5
  p2 = 7

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())

  # Train MB-sPLS *without* storing raw training blocks.
  po_mbspls = mlr3mbspls::PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      store_train_blocks = FALSE,
      append = FALSE
    )
  )

  task_lv = po_mbspls$train(list(task))[[1]]
  # At this stage, block features are no longer present in the task backend.

  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = FALSE,
      B = 3L,
      workers = 1L
    )
  )

  expect_error(
    po_sel$train(list(task_lv)),
    "Cannot rebuild training blocks"
  )
})
