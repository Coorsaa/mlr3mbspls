library(testthat)

set.seed(123)

for (n in c(50, 200)) {
  for (p in c(5, 15)) {

    X = cbind(1, matrix(rnorm(n * p), n)) # include intercept
    B = matrix(rnorm((p + 1) * 3), p + 1) # 3 responses
    Y = X %*% B + matrix(rnorm(n * 3, 0, 0.1), n)

    # R reference via qr.solve
    ref = qr.coef(qr(X), Y)

    # C++ version
    cpp = cpp_lm_coeff(X, Y)

    test_that(sprintf("coefficients match (n=%d, p=%d)", n, p), {
      expect_equal(cpp, ref, tolerance = 1e-8)
    })
  }
}


test_that("native ridge rejects invalid unpenalized indices", {
  design = cbind(1, 1:4)
  response = matrix(c(1, 3, 4, 8), ncol = 1)
  for (index in c(0L, -1L, 3L, NA_integer_)) {
    expect_error(cpp_lm_coeff_ridge(design, response, 1, index),
      "valid one-based column indices")
  }
})


test_that("native least squares does not silently approximate singular designs", {
  design = cbind(1, 1:4, 0)
  response = matrix(c(1, 3, 4, 8), ncol = 1)
  expect_error(cpp_lm_coeff(design, response), "triangular solve failed")
  expect_error(cpp_lm_coeff(design[1:2, ], response[1:2, , drop = FALSE]),
    "at least as many rows as columns")
})
