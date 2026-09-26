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
  expect_false(".Y_aux" %in% po$state$target_columns)
})

test_that("PipeOpMBsPLSXY rejects components beyond target rank", {
  set.seed(13)
  classification_task = mlr3::TaskClassif$new(
    id = "mbxy_rank_classif",
    backend = data.frame(
      x1 = rnorm(30L),
      x2 = rnorm(30L),
      y = factor(rep(c("A", "B"), 15L))
    ),
    target = "y"
  )
  regression_task = mlr3::TaskRegr$new(
    id = "mbxy_rank_regr",
    backend = data.frame(
      x1 = rnorm(30L),
      x2 = rnorm(30L),
      y = rnorm(30L)
    ),
    target = "y"
  )

  for (task in list(classification_task, regression_task)) {
    pipeop = PipeOpMBsPLSXY$new(
      blocks = list(block = c("x1", "x2")),
      param_vals = list(ncomp = 2L, y_rep = 3L)
    )
    expect_error(
      pipeop$train(list(task)),
      "effective rank 1"
    )
  }
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
  expect_equal(dim(po$state$c_matrix), c(2L, 1L))
  expect_equal(rownames(po$state$c_matrix), c("b1", ".target"))
  expect_lte(po$state$c_matrix[".target", 1L], sqrt(2))
})


test_that("PipeOpMBsPLSXY rejects malformed or infeasible c_matrix values", {
  set.seed(13)

  dt = data.table::data.table(
    x1 = rnorm(36),
    x2 = rnorm(36),
    y = factor(sample(c("A", "B"), 36, replace = TRUE))
  )
  task = mlr3::TaskClassif$new(id = "mbxy_bad_cmat", backend = dt, target = "y")
  blocks = list(b1 = c("x1", "x2"))

  duplicate_rows = matrix(
    1,
    nrow = 2L,
    ncol = 1L,
    dimnames = list(c("b1", "b1"), "LC1")
  )
  expect_error(
    PipeOpMBsPLSXY$new(
      blocks = blocks,
      param_vals = list(c_matrix = duplicate_rows)
    ),
    "row names must be unique"
  )

  duplicate_columns = matrix(
    1,
    nrow = 1L,
    ncol = 2L,
    dimnames = list("b1", c("LC1", "LC1"))
  )
  expect_error(
    PipeOpMBsPLSXY$new(
      blocks = blocks,
      param_vals = list(c_matrix = duplicate_columns)
    ),
    "column names must be unique"
  )

  bad_x_budget = matrix(
    2,
    nrow = 1L,
    ncol = 1L,
    dimnames = list("b1", "LC1")
  )
  po_x = PipeOpMBsPLSXY$new(
    blocks = blocks,
    param_vals = list(c_matrix = bad_x_budget)
  )
  expect_error(
    po_x$train(list(task)),
    "sparsity budget.*sqrt\\(p_block\\)"
  )

  bad_target_budget = matrix(
    c(1, 2),
    nrow = 2L,
    ncol = 1L,
    dimnames = list(c("b1", ".target"), "LC1")
  )
  po_target = PipeOpMBsPLSXY$new(
    blocks = blocks,
    param_vals = list(c_matrix = bad_target_budget)
  )
  expect_error(
    po_target$train(list(task)),
    "sparsity budget.*sqrt\\(p_block\\)"
  )
})
