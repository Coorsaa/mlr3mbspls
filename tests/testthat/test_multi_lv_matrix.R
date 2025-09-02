test_that("cpp_mbspls_multi_lv_cmatrix replicates C++ output for equal C", {
  X = list(matrix(rnorm(100 * 5), 100), matrix(rnorm(100 * 7), 100))
  Cvec = rep(sqrt(5), 2)
  Cmat = matrix(Cvec, 2, 3)
  r1 = cpp_mbspls_multi_lv_cmatrix(X, Cmat, do_perm = FALSE)
  r2 = cpp_mbspls_multi_lv(X, Cvec, K = 3, do_perm = FALSE)
  expect_equal(r1$objective, r2$objective, tolerance = 1e-8)
})
