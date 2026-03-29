test_that("PipeOpBlockScaling preserves supervised targets on train and predict", {
  task = task_multiblock_synthetic(task_type = "classif", n = 30L, seed = 55L)
  blocks = task$block_features()

  po = PipeOpBlockScaling$new(
    param_vals = list(
      blocks = blocks,
      method = "unit_ssq"
    )
  )

  out_train = po$train(list(task))[[1L]]
  expect_true(inherits(out_train, "TaskClassif"))
  expect_equal(out_train$target_names, task$target_names)
  expect_true(all(task$target_names %in% out_train$col_info$id))
  expect_equal(out_train$class_names, task$class_names)

  out_pred = po$predict(list(task))[[1L]]
  expect_true(inherits(out_pred, "TaskClassif"))
  expect_equal(out_pred$target_names, task$target_names)
  expect_true(all(task$target_names %in% out_pred$col_info$id))
  expect_equal(out_pred$class_names, task$class_names)
})


test_that("PipeOpBlockScaling preserves features that collide with the internal row-id name", {
  n = 24L
  df = data.frame(
    `..row_id_blockscale` = rnorm(n),
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "blockscale_collision", backend = df, target = "y")

  po = PipeOpBlockScaling$new(
    param_vals = list(
      blocks = list(b1 = c("..row_id_blockscale", "x1", "x2")),
      method = "unit_ssq"
    )
  )

  out = po$train(list(task))[[1L]]
  expect_true("..row_id_blockscale" %in% out$feature_names)
})
