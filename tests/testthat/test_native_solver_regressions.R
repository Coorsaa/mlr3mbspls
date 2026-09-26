test_that("native sparse solvers respect L1 budgets with tied gradients", {
  z = as.numeric(scale(seq_len(12L)))
  block = cbind(z, -z, z)
  for (budget in c(1, 1.2, 1.5)) {
    set.seed(62L)
    pls = mlr3mbspls:::cpp_mbspls_one_lv(
      list(block, block), rep(budget, 2L), max_iter = 20L, tol = 1e-8
    )
    pca = mlr3mbspls:::cpp_mbspca_one_lv(
      list(block), budget, max_iter = 20L, tol = 1e-8
    )
    for (weights in c(pls$W, pca$W)) {
      expect_equal(sqrt(sum(weights^2)), 1, tolerance = 1e-8)
      expect_equal(sum(abs(weights)), budget, tolerance = 1e-8)
      # The linear update attains the L1 upper bound on tied maxima.
      expect_equal(abs(sum(c(1, -1, 1) * weights)), budget,
        tolerance = 1e-8)
    }
  }
})

test_that("PCA initialization does not cancel anticorrelated block scores", {
  z = as.numeric(scale(seq_len(12L)))
  block = cbind(z, 2 * z)
  fit = mlr3mbspls:::cpp_mbspca_one_lv(
    list(block, -block), rep(sqrt(2), 2L), max_iter = 20L
  )
  scores = cbind(block %*% fit$W[[1L]], -block %*% fit$W[[2L]])
  expect_equal(unname(cor(scores)[1L, 2L]), 1, tolerance = 1e-8)
  expect_true(fit$converged)
})

test_that("native PCA permutations count numerically tied statistics", {
  block = cbind(seq_len(12L), seq_len(12L)^2)
  fit = mlr3mbspls:::cpp_mbspca_one_lv(list(block), sqrt(2))
  set.seed(64L)
  # One-block row permutations preserve the PCA problem exactly.
  p = mlr3mbspls:::perm_test_component_mbspca(
    list(block), fit$W, sqrt(2), n_perm = 19L
  )
  expect_equal(p, 1)
})

test_that("bootstrap replicates preserve the observed set of block pairs", {
  # The third block is constant in resamples that omit its sole positive row.
  # Those draws must fail instead of becoming a two-block statistic.
  x = list(matrix(1:6), matrix(c(3, 1, 5, 2, 6, 4)),
    matrix(c(1, rep(0, 5))))
  set.seed(65L)
  result = mlr3mbspls:::cpp_bootstrap_test_oos(
    x, list(1, 1, 1), n_boot = 100L
  )
  expect_gt(result$n_boot_failed, 0L)
  expect_equal(result$n_boot + result$n_boot_failed, 100L)
  expect_length(result$replicates, result$n_boot)
})
