test_that("PipeOpMBsPLSXY trains on classification tasks without levels shadowing", {
  set.seed(11)

  dt = data.table::data.table(
    x1 = rnorm(40),
    x2 = rnorm(40),
    x3 = rnorm(40),
    y = factor(sample(c("A", "B"), 40, replace = TRUE), levels = c("A", "B"))
  )

  task = mlr3::TaskClassif$new(id = "mbxy_train", backend = dt, target = "y")
  po = PipeOpMBsPLSXY$new(
    blocks = list(b1 = c("x1", "x2", "x3")),
    param_vals = list(ncomp = 1L, emit_y_scores = TRUE)
  )

  out = po$train(list(task))[[1L]]

  expect_s3_class(out, "Task")
  expect_true(all(c("LV1_b1", "LV1_.Y") %in% out$feature_names))
  expect_true(is.list(po$state$weights_x))
  expect_true(is.list(po$state$weights_y))
})



test_that("PipeOpMBsPLSXY accepts rownamed c_matrix without explicit .target row", {
  set.seed(12)

  dt = data.table::data.table(
    x1 = rnorm(36),
    x2 = rnorm(36),
    x3 = rnorm(36),
    y = factor(sample(c("A", "B"), 36, replace = TRUE), levels = c("A", "B"))
  )

  task = mlr3::TaskClassif$new(id = "mbxy_cmat", backend = dt, target = "y")
  cm = matrix(1.5, nrow = 1L, ncol = 1L, dimnames = list("b1", "comp1"))

  po = PipeOpMBsPLSXY$new(
    blocks = list(b1 = c("x1", "x2", "x3")),
    param_vals = list(c_matrix = cm, c_target = 2, emit_y_scores = FALSE)
  )

  out = po$train(list(task))[[1L]]

  expect_s3_class(out, "Task")
  expect_true("LV1_b1" %in% out$feature_names)
})
