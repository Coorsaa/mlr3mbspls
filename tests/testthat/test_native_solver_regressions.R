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

test_that("native PCA permutation diagnostic requires two blocks", {
  block = cbind(seq_len(12L), seq_len(12L)^2)
  fit = mlr3mbspls:::cpp_mbspca_one_lv(list(block), sqrt(2))
  set.seed(64L)
  # One-block row permutations preserve the PCA problem exactly, so the
  # cross-block null is undefined rather than a p-value of one.
  expect_error(
    mlr3mbspls:::perm_test_component_mbspca(
      list(block), fit$W, sqrt(2), n_perm = 19L
    ),
    "at least two blocks"
  )

  set.seed(64L)
  blocks = list(block, cbind(rev(seq_len(12L)), rnorm(12L)))
  fit = mlr3mbspls:::cpp_mbspca_one_lv(blocks, rep(sqrt(2), 2L))
  p = mlr3mbspls:::perm_test_component_mbspca(
    blocks, fit$W, rep(sqrt(2), 2L), n_perm = 19L
  )
  expect_true(p > 0 && p <= 1)
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

# ── MB-sPLS one-LV solver: Gauss-Seidel sweeps and deterministic start ──────

# Two blocks sharing two latent factors (five features each) plus ten noise
# features. With c = 1.3 the former simultaneous (Jacobi) sweeps fell into
# period-2 cycles for most random starts on this data.
mbspls_two_factor_blocks = function(seed = 1L, n = 80L) {
  set.seed(seed)
  f1 = rnorm(n)
  f2 = rnorm(n)
  lapply(1:2, function(b) {
    x = matrix(rnorm(n * 20L), n, 20L)
    x[, 1:5] = x[, 1:5] + f1
    x[, 6:10] = x[, 6:10] + f2
    scale(x)
  })
}

mbspls_synthetic_blocks = function(n = 90L, seed = 1L) {
  task = task_multiblock_synthetic(n = n, seed = seed)
  lapply(task$block_features(), function(cols) {
    scale(as.matrix(task$data(cols = cols)))
  })
}

# Reference PMD projection: max g'w subject to ||w||_2 = 1, ||w||_1 <= budget.
mbspls_pmd_project = function(g, budget) {
  w = g / sqrt(sum(g^2))
  if (sum(abs(w)) <= budget) {
    return(w)
  }
  soft = function(delta) {
    s = sign(g) * pmax(abs(g) - delta, 0)
    s / sqrt(sum(s^2))
  }
  delta = stats::uniroot(
    function(delta) sum(abs(soft(delta))) - budget,
    c(0, max(abs(g)) * (1 - 1e-12)),
    tol = 1e-14
  )$root
  soft(delta)
}

# One block-coordinate sweep with the solver's target: the mean z-scored
# score of the other blocks at their latest values.
mbspls_gauss_seidel_sweep = function(blocks, weights, budgets) {
  n = nrow(blocks[[1L]])
  for (b in seq_along(blocks)) {
    others = setdiff(seq_along(blocks), b)
    z = vapply(others, function(k) {
      score = drop(blocks[[k]] %*% weights[[k]])
      (score - mean(score)) / stats::sd(score)
    }, numeric(n))
    target = rowMeans(matrix(z, nrow = n))
    weights[[b]] = mbspls_pmd_project(drop(crossprod(blocks[[b]], target)), budgets[[b]])
  }
  weights
}

# Correlation of each block score with the mean z-scored score of the others.
mbspls_consensus_correlations = function(blocks, weights) {
  scores = vapply(seq_along(blocks), function(b) {
    drop(blocks[[b]] %*% weights[[b]])
  }, numeric(nrow(blocks[[1L]])))
  z = scale(scores)
  vapply(seq_along(blocks), function(b) {
    stats::cor(scores[, b], rowMeans(z[, -b, drop = FALSE]))
  }, numeric(1L))
}

test_that("MB-sPLS fits do not depend on the RNG state", {
  cases = list(
    list(blocks = mbspls_two_factor_blocks(), budgets = c(1.3, 1.3)),
    list(blocks = mbspls_synthetic_blocks(), budgets = rep(sqrt(6), 3L))
  )
  for (case in cases) {
    set.seed(1L)
    first = mlr3mbspls:::cpp_mbspls_one_lv(case$blocks, case$budgets, 600L, 1e-4)
    set.seed(2L)
    state = .Random.seed
    second = mlr3mbspls:::cpp_mbspls_one_lv(case$blocks, case$budgets, 600L, 1e-4)
    expect_identical(.Random.seed, state)
    expect_identical(first, second)

    set.seed(3L)
    multi_first = mlr3mbspls:::cpp_mbspls_multi_lv(case$blocks, case$budgets, K = 2L)
    set.seed(4L)
    multi_second = mlr3mbspls:::cpp_mbspls_multi_lv(case$blocks, case$budgets, K = 2L)
    expect_identical(multi_first, multi_second)
  }
})

test_that("MB-sPLS fits are invariant to the order of the samples", {
  blocks = mbspls_synthetic_blocks()
  budgets = c(1.5, 1.5, 1.5)
  order = rev(seq_len(nrow(blocks[[1L]])))
  set.seed(1L)
  fit = mlr3mbspls:::cpp_mbspls_one_lv(blocks, budgets, 600L, 1e-4)
  set.seed(2L)
  permuted = mlr3mbspls:::cpp_mbspls_one_lv(
    lapply(blocks, function(x) x[order, , drop = FALSE]), budgets, 600L, 1e-4
  )
  expect_equal(permuted$W, fit$W, tolerance = 1e-8)
  expect_equal(permuted$objective, fit$objective, tolerance = 1e-10)
})

test_that("MB-sPLS converges where simultaneous block updates cycled", {
  blocks = mbspls_two_factor_blocks()
  for (seed in 1:5) {
    set.seed(seed)
    fit = mlr3mbspls:::cpp_mbspls_one_lv(blocks, c(1.3, 1.3), 600L, 1e-4)
    expect_true(fit$converged)
    expect_lt(fit$iterations, 600L)
    # The best objective over 50 random starts with tol = 1e-8; mismatched
    # period-2 pairs returned objectives between 0.02 and 0.71 here.
    expect_equal(fit$objective, 0.7567658, tolerance = 1e-6)
  }
  # The fit no longer depends on the parity of max_iter.
  longer = mlr3mbspls:::cpp_mbspls_one_lv(blocks, c(1.3, 1.3), 601L, 1e-4)
  expect_identical(longer$W, fit$W)
})

test_that("MB-sPLS returns a fixed point of the block-coordinate update", {
  cases = list(
    list(blocks = mbspls_two_factor_blocks(), budgets = c(1.3, 1.3)),
    list(blocks = mbspls_two_factor_blocks(), budgets = c(2, 3)),
    list(blocks = mbspls_synthetic_blocks(), budgets = c(1.2, 1.5, 2))
  )
  tol = 1e-4
  for (case in cases) {
    for (seed in 1:3) {
      set.seed(seed)
      fit = mlr3mbspls:::cpp_mbspls_one_lv(case$blocks, case$budgets, 600L, tol)
      expect_true(fit$converged)
      swept = mbspls_gauss_seidel_sweep(case$blocks, fit$W, case$budgets)
      change = max(mapply(function(new, old) sqrt(sum((new - old)^2)), swept, fit$W))
      expect_lt(change, tol)
    }
  }
})

test_that("MB-sPLS block scores are sign-coherent with a canonical global sign", {
  cases = list(
    list(blocks = mbspls_two_factor_blocks(), budgets = c(1.3, 1.3)),
    list(blocks = mbspls_two_factor_blocks(3L), budgets = rep(sqrt(20), 2L)),
    list(blocks = mbspls_synthetic_blocks(), budgets = rep(sqrt(6), 3L)),
    list(blocks = mbspls_synthetic_blocks(50L, 2L), budgets = c(1, 1.5, 2))
  )
  for (case in cases) {
    for (seed in 1:5) {
      set.seed(seed)
      fit = mlr3mbspls:::cpp_mbspls_one_lv(case$blocks, case$budgets, 600L, 1e-4)
      expect_true(all(mbspls_consensus_correlations(case$blocks, fit$W) > 0))
      lead = fit$W[[1L]][which.max(abs(fit$W[[1L]]))]
      expect_gt(lead, 0)
    }
  }
})

test_that("MB-sPLS convergence does not depend on the reported criterion", {
  for (data_seed in 1:10) {
    set.seed(data_seed)
    n = 15L
    z = rnorm(n)
    blocks = lapply(1:2, function(b) {
      x = matrix(rnorm(n * 6L), n, 6L)
      x[, 1:3] = x[, 1:3] + z
      scale(x)
    })
    budgets = rep(sqrt(6), 2L)
    pearson = mlr3mbspls:::cpp_mbspls_one_lv(blocks, budgets, 600L, 1e-4)
    spearman = mlr3mbspls:::cpp_mbspls_one_lv(blocks, budgets, 600L, 1e-4,
      spearman = TRUE)
    # Rank correlations are piecewise constant in the weights; stopping on
    # them used to end the iteration while the weights were still moving.
    expect_identical(spearman$W, pearson$W)
    expect_identical(spearman$iterations, pearson$iterations)
    scores = mapply(function(x, w) drop(x %*% w), blocks, spearman$W)
    expect_equal(spearman$objective,
      abs(stats::cor(scores[, 1L], scores[, 2L], method = "spearman")),
      tolerance = 1e-12)
  }
})

test_that("MB-sPLS solvers report convergence per kept component", {
  blocks = mbspls_synthetic_blocks()
  budgets = rep(sqrt(6), 3L)

  one = mlr3mbspls:::cpp_mbspls_one_lv(blocks, budgets, 600L, 1e-4)
  expect_true(is.logical(one$converged) && length(one$converged) == 1L)
  expect_true(is.integer(one$iterations) && length(one$iterations) == 1L)
  expect_true(one$iterations >= 1L && one$iterations <= 600L)

  capped = mlr3mbspls:::cpp_mbspls_one_lv(blocks, c(1.2, 1.2, 1.2), 1L, 1e-12)
  expect_false(capped$converged)
  expect_identical(capped$iterations, 1L)

  vector_fit = mlr3mbspls:::cpp_mbspls_multi_lv(blocks, budgets, K = 3L)
  matrix_fit = mlr3mbspls:::cpp_mbspls_multi_lv_cmatrix(
    blocks, matrix(budgets, 3L, 3L)
  )
  for (fit in list(vector_fit, matrix_fit)) {
    expect_true(is.logical(fit$converged))
    expect_true(is.integer(fit$iterations))
    expect_length(fit$converged, length(fit$objective))
    expect_length(fit$iterations, length(fit$objective))
    expect_true(all(fit$converged))
  }

  set.seed(7L)
  n = 60L
  signal = rnorm(n)
  one_factor = lapply(1:2, function(b) {
    x = matrix(rnorm(n * 5L), n, 5L)
    x[, 1:2] = x[, 1:2] + 2 * signal
    scale(x)
  })
  set.seed(8L)
  stopped = mlr3mbspls:::cpp_mbspls_multi_lv(one_factor, rep(sqrt(5), 2L),
    K = 3L, do_perm = TRUE, n_perm = 49L)
  expect_lt(length(stopped$objective), 3L)
  expect_length(stopped$converged, length(stopped$objective))
  expect_length(stopped$iterations, length(stopped$objective))

  capped_multi = mlr3mbspls:::cpp_mbspls_multi_lv(blocks, c(1.2, 1.2, 1.2),
    K = 2L, max_iter = 1L, tol = 1e-12)
  expect_identical(capped_multi$converged, c(FALSE, FALSE))
  expect_identical(capped_multi$iterations, c(1L, 1L))
})

test_that("MB-sPLS permutation screening refits with the observed-fit settings", {
  blocks = mbspls_two_factor_blocks()
  set.seed(9L)
  with_perm = mlr3mbspls:::cpp_mbspls_multi_lv(blocks, c(1.3, 1.3), K = 2L,
    do_perm = TRUE, n_perm = 19L)
  without = mlr3mbspls:::cpp_mbspls_multi_lv(blocks, c(1.3, 1.3), K = 2L)
  kept = seq_along(with_perm$W)
  # The permutations draw from the RNG; the fitted components do not.
  expect_identical(with_perm$W, without$W[kept])
  expect_identical(with_perm$objective, without$objective[kept])
  expect_lte(with_perm$p_values[[1L]], 0.05)
})

test_that("MB-sPLS keeps coerced integer blocks alive during the fit", {
  skip_on_cran()
  set.seed(5L)
  n = 60L
  signal = rnorm(n)
  integer_blocks = lapply(1:8, function(b) {
    x = matrix(rpois(n * 40L, 3), n, 40L)
    x[, 1:5] = x[, 1:5] + round(2 * signal)
    storage.mode(x) = "integer"
    x
  })
  double_blocks = lapply(integer_blocks, function(x) {
    storage.mode(x) = "double"
    x
  })
  budgets = rep(2, 8L)
  reference = mlr3mbspls:::cpp_mbspls_one_lv(double_blocks, budgets, 50L, 1e-4)
  # Force collections while the coerced copies would otherwise be released.
  gctorture(TRUE)
  on.exit(gctorture(FALSE), add = TRUE)
  fit = mlr3mbspls:::cpp_mbspls_one_lv(integer_blocks, budgets, 50L, 1e-4)
  gctorture(FALSE)
  expect_identical(fit, reference)
})

test_that("unused legacy native helpers are not registered", {
  ns = asNamespace("mlr3mbspls")
  for (name in c("cpp_ev_test", "cpp_mbspls_bootstrap",
    "cpp_bootstrap_latent_correlation")) {
    expect_false(exists(name, envir = ns, inherits = FALSE))
  }
})

test_that("test-set explained variance removes the fitted rank-one term", {
  set.seed(11L)
  train = mbspls_synthetic_blocks(60L, 3L)
  test = lapply(mbspls_synthetic_blocks(40L, 4L), unname)
  fit = mlr3mbspls:::cpp_mbspls_multi_lv(train, rep(sqrt(6), 3L), K = 1L)
  ev = mlr3mbspls:::cpp_compute_test_ev_core(test, fit$W, fit$P,
    use_train_loadings = TRUE)
  expected = vapply(seq_along(test), function(b) {
    score = drop(test[[b]] %*% fit$W[[1L]][[b]])
    residual = test[[b]] - tcrossprod(score, fit$P[[1L]][[b]])
    (sum(test[[b]]^2) - sum(residual^2)) / sum(test[[b]]^2)
  }, numeric(1L))
  expect_equal(drop(ev$ev_block[1L, ]), expected, tolerance = 1e-10)
})
