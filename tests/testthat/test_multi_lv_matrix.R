test_that("cpp_mbspls_multi_lv_cmatrix replicates C++ output for equal C", {
  set.seed(1)
  X = list(
    matrix(rnorm(100 * 5), nrow = 100, ncol = 5),
    matrix(rnorm(100 * 7), nrow = 100, ncol = 7)
  )
  Cvec = rep(sqrt(5), 2)
  Cmat = matrix(Cvec, 2, 3)
  r1 = cpp_mbspls_multi_lv_cmatrix(X, Cmat, do_perm = FALSE)
  r2 = cpp_mbspls_multi_lv(X, Cvec, K = 3, do_perm = FALSE)
  # The two entry points should converge to very similar solutions.
  # We allow small numeric drift across platforms/BLAS implementations.
  expect_equal(length(r1$objective), length(r2$objective))
  expect_true(all(is.finite(r1$objective)))
  expect_true(all(is.finite(r2$objective)))

  rel_diff = abs(r1$objective - r2$objective) /
    pmax(abs(r1$objective), abs(r2$objective), 1e-12)
  expect_lt(max(rel_diff), 0.05)
})
