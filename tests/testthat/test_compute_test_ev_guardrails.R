test_that("compute_test_ev requires explicit test_ls when training loadings are absent", {
  x1 = matrix(rnorm(20), nrow = 10, ncol = 2)
  x2 = matrix(rnorm(20), nrow = 10, ncol = 2)
  colnames(x1) = c("a1", "a2")
  colnames(x2) = c("b1", "b2")

  W_all = list(list(
    block_a = c(a1 = 1, a2 = 0),
    block_b = c(b1 = 0, b2 = 1)
  ))

  expect_error(
    compute_test_ev(
      X_blocks_test = list(block_a = x1, block_b = x2),
      W_all = W_all
    ),
    "loading_source='auto' requires training loadings"
  )
})
