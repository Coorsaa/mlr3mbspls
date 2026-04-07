test_that("PipeOpMBsPLSXY safely scales constant class-indicator columns", {
  po = PipeOpMBsPLSXY$new(
    blocks = list(b1 = c("x1", "x2")),
    param_vals = list(ncomp = 1L)
  )

  y_mat = po$.__enclos_env__$private$.build_y_matrix(
    task = NULL,
    target_vec = factor(rep("A", 6), levels = c("A", "B", "C")),
    levs = c("A", "B", "C"),
    center = TRUE,
    scale = TRUE
  )

  expect_true(is.matrix(y_mat))
  expect_equal(dim(y_mat), c(6L, 3L))
  expect_true(all(is.finite(y_mat)))
  expect_equal(as.numeric(y_mat[, 2L]), rep(0, 6))
  expect_equal(as.numeric(y_mat[, 3L]), rep(0, 6))
})
