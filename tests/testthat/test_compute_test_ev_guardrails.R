test_that("mac_comp is NaN (not 0) when all test block scores have zero variance", {
  # When every column of every test block is constant, Pearson correlations are
  # undefined (n_pairs == 0 in C++).  The result must be NaN so downstream
  # na.rm=TRUE summaries exclude it rather than treating missing correlation as 0.
  x_const = matrix(1, nrow = 5, ncol = 2)
  colnames(x_const) = c("a1", "a2")
  colnames(x_const) = c("a1", "a2")   # both blocks constant

  W_all = list(list(
    block_a = c(a1 = 0.5, a2 = 0.5),
    block_b = c(b1 = 0.5, b2 = 0.5)
  ))
  x_const_b = x_const
  colnames(x_const_b) = c("b1", "b2")

  P_all = list(list(
    block_a = c(a1 = 1, a2 = 0),
    block_b = c(b1 = 1, b2 = 0)
  ))

  res = compute_test_ev(
    X_blocks_test = list(block_a = x_const, block_b = x_const_b),
    W_all = W_all,
    P_all = P_all,
    loading_source = "train"
  )

  # mac_comp must be NaN, not 0, so EV-weighted aggregation can skip it safely
  expect_true(is.nan(res$mac_comp[[1L]]),
    info = "mac_comp should be NaN when all block scores are degenerate (zero-variance)")
  expect_true(is.na(mean(res$mac_comp, na.rm = FALSE)),
    info = "mean mac without na.rm should propagate the NaN")
  expect_equal(length(res$mac_comp), 1L)
})

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
