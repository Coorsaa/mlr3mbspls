test_that("multi-block PipeOps hash their blocks without mlr3pipelines warnings", {
  blocks_1 = list(a = c("x1", "x2"), b = c("z1", "z2"))
  blocks_2 = list(a = c("x1", "x2", "x3"), b = c("z1", "z2"))
  constructors = list(PipeOpMBsPLS, PipeOpMBsPLSXY, PipeOpMBsPCA)

  for (constructor in constructors) {
    first = constructor$new(blocks = blocks_1)
    expect_no_warning(first$phash)
    expect_identical(first$phash, constructor$new(blocks = blocks_1)$phash)
    expect_false(identical(first$phash, constructor$new(blocks = blocks_2)$phash))
  }

  # phash identifies the operator independently of hyperparameter values.
  tuned = PipeOpMBsPLS$new(blocks = blocks_1, param_vals = list(append = TRUE, seed_train = 1L))
  expect_identical(tuned$phash, PipeOpMBsPLS$new(blocks = blocks_1)$phash)
})
