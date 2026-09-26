test_that("cmatrix solver with equal columns matches the vector solver exactly", {
  set.seed(1)
  X = list(
    matrix(rnorm(100 * 5), nrow = 100, ncol = 5),
    matrix(rnorm(100 * 7), nrow = 100, ncol = 7)
  )
  Cvec = rep(sqrt(5), 2)
  Cmat = matrix(Cvec, 2, 3)
  # Both entry points run the same deterministic one-LV solver, so identical
  # inputs must give identical outputs, whatever the RNG state.
  set.seed(11)
  r1 = cpp_mbspls_multi_lv_cmatrix(X, Cmat, do_perm = FALSE)
  set.seed(12)
  r2 = cpp_mbspls_multi_lv(X, Cvec, K = 3, do_perm = FALSE)
  expect_identical(r1, r2)
  expect_equal(dim(r1$T_mat), c(100L, 6L))
  expect_true(all(is.finite(r1$objective)))

  # With permutation screening the permutations draw from the RNG, so the
  # same RNG state must reproduce p-values and the early-stopping decision.
  set.seed(13)
  p1 = cpp_mbspls_multi_lv_cmatrix(X, Cmat, do_perm = TRUE, n_perm = 19L)
  set.seed(13)
  p2 = cpp_mbspls_multi_lv(X, Cvec, K = 3, do_perm = TRUE, n_perm = 19L)
  expect_identical(p1, p2)
})


test_that("multi-component solvers do not deflate after the final component", {
  set.seed(2)
  n = 40L
  x = matrix(rnorm(n * 3L), nrow = n)
  y = matrix(scale(rnorm(n)), ncol = 1L)
  blocks = list(x, y)
  constraints = c(sqrt(3), 1)

  fit_vector = cpp_mbspls_multi_lv(
    blocks,
    constraints,
    K = 1L,
    do_perm = FALSE
  )
  fit_matrix = cpp_mbspls_multi_lv_cmatrix(
    blocks,
    matrix(constraints, nrow = 2L),
    do_perm = FALSE
  )

  expect_length(fit_vector$W, 1L)
  expect_length(fit_matrix$W, 1L)
  expect_equal(dim(fit_vector$T_mat), c(n, 2L))
  expect_equal(dim(fit_matrix$T_mat), c(n, 2L))
})


test_that("vector solver passes the Spearman objective to one-LV fitting", {
  set.seed(3)
  n = 60L
  z = rnorm(n)
  blocks = list(
    cbind(z, z^2, rnorm(n)),
    cbind(exp(z / 2), z^3, rnorm(n))
  )
  constraints = c(sqrt(3), sqrt(3))

  set.seed(31)
  direct = cpp_mbspls_one_lv(
    blocks,
    constraints,
    max_iter = 500L,
    tol = 1e-4,
    spearman = TRUE
  )
  set.seed(31)
  multi = cpp_mbspls_multi_lv(
    blocks,
    constraints,
    K = 1L,
    max_iter = 500L,
    tol = 1e-4,
    spearman = TRUE
  )

  expect_equal(multi$objective[[1L]], direct$objective, tolerance = 1e-12)
  expect_equal(multi$W[[1L]], direct$W, tolerance = 1e-12)
})
