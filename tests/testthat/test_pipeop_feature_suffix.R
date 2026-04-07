test_that("PipeOpFeatureSuffix renames features without false-positive collisions", {
  df = data.frame(
    x = 1:4,
    x_sfx = 5:8,
    y = c(1, 2, 3, 4)
  )
  task = mlr3::TaskRegr$new(id = "suffix_collision", backend = df, target = "y")

  po = PipeOpFeatureSuffix$new(param_vals = list(
    suffix = "_sfx",
    error_on_collision = TRUE,
    skip_already_suffixed = FALSE
  ))

  out = po$train(list(task))[[1L]]
  expect_setequal(out$feature_names, c("x_sfx", "x_sfx_sfx"))
})



test_that("PipeOpFeatureSuffix keeps TaskMultiBlock block metadata synchronized", {
  dt = data.table::data.table(
    a1 = 1:4,
    b1 = 5:8,
    y = factor(c("a", "b", "a", "b"))
  )
  task = TaskMultiBlock(dt, blocks = list(a = "a1", b = "b1"), target = "y", task_type = "classif")

  po = PipeOpFeatureSuffix$new(param_vals = list(suffix = "_sfx"))
  out = po$train(list(task))[[1L]]

  expect_equal(out$blocks, list(a = "a1_sfx", b = "b1_sfx"))
  expect_true(all(c("a1_sfx", "b1_sfx") %in% out$feature_names))
})
