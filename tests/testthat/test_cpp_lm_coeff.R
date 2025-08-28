library(testthat)

set.seed(123)

for (n in c(50, 200)) {
  for (p in c(5, 15)) {

    X <- cbind(1, matrix(rnorm(n * p), n))      # include intercept
    B <- matrix(rnorm((p + 1) * 3), p + 1)      # 3 responses
    Y <- X %*% B + matrix(rnorm(n * 3, 0, 0.1), n)

    # R reference via qr.solve
    ref  <- qr.coef(qr(X), Y)

    # C++ version
    cpp  <- cpp_lm_coeff(X, Y)

    test_that(sprintf("coefficients match (n=%d, p=%d)", n, p), {
      expect_equal(cpp, ref, tolerance = 1e-8)
    })
  }
}
