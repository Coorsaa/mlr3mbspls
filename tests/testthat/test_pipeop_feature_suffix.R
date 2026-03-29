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
