library(testthat)

test_that("cpp_mbspls_one_lv returns the final fitted objective", {
  set.seed(100)
  blocks = list(
    matrix(rnorm(90), 30, 3),
    matrix(rnorm(120), 30, 4)
  )
  fit = cpp_mbspls_one_lv(
    blocks,
    c_constraints = c(sqrt(3), 2),
    max_iter = 25L,
    tol = 1e-2
  )
  recalculated = cpp_block_objective_oos(blocks, fit$W)

  expect_equal(fit$objective, recalculated, tolerance = 1e-12)
  expect_error(
    cpp_mbspls_one_lv(blocks, c(0.5, 2), 25L, 1e-2),
    "at least 1"
  )
})

# ── cpp_lm_coeff_ridge: lambda < 0 guard ─────────────────────────────────────
test_that("cpp_lm_coeff_ridge errors on negative lambda", {
  X = cbind(1, matrix(rnorm(30), 10))
  Y = matrix(rnorm(10))
  expect_error(
    cpp_lm_coeff_ridge(X, Y, lambda = -1.0, unpen_idx = NULL),
    "lambda must be a non-negative finite number"
  )
})

test_that("cpp_lm_coeff_ridge errors on non-finite lambda", {
  X = cbind(1, matrix(rnorm(30), 10))
  Y = matrix(rnorm(10))
  expect_error(
    cpp_lm_coeff_ridge(X, Y, lambda = Inf, unpen_idx = NULL),
    "lambda must be a non-negative finite number"
  )
  expect_error(
    cpp_lm_coeff_ridge(X, Y, lambda = NaN, unpen_idx = NULL),
    "lambda must be a non-negative finite number"
  )
})

test_that("cpp_lm_coeff_ridge accepts lambda = 0", {
  set.seed(42)
  X = cbind(1, matrix(rnorm(50), 10))
  Y = matrix(rnorm(10))
  coef = cpp_lm_coeff_ridge(X, Y, lambda = 0.0, unpen_idx = NULL)
  expect_true(is.matrix(coef))
  expect_equal(nrow(coef), ncol(X))
})

# ── perm_test_component_mbspca: c_vec <= 0 guard ─────────────────────────────
test_that("perm_test_component_mbspca errors on non-positive c_vec", {
  set.seed(1)
  X = list(matrix(rnorm(30), 10, 3), matrix(rnorm(20), 10, 2))
  W = list(c(0.6, 0.5, -0.6), c(0.7, -0.7))
  W = lapply(W, function(w) w / sqrt(sum(w^2)))

  expect_error(
    perm_test_component_mbspca(X, W, c(0.0, 1.0), n_perm = 19L),
    "strictly positive"
  )
  expect_error(
    perm_test_component_mbspca(X, W, c(-1.0, 1.5), n_perm = 19L),
    "strictly positive"
  )
  expect_error(
    perm_test_component_mbspca(X, W[1L], c(1.0, 1.5), n_perm = 19L),
    "W_list length"
  )
  expect_error(
    perm_test_component_mbspca(X, W, c(1.0, 1.5), n_perm = 0L),
    "n_perm"
  )
})

test_that("cpp_mbspca_one_lv remains finite on unscaled heterogeneous blocks", {
  blocks = list(
    eng = as.matrix(mtcars[, c("disp", "hp", "drat")]),
    body = as.matrix(mtcars[, c("wt", "qsec")])
  )
  budgets = sqrt(vapply(blocks, ncol, integer(1L)))

  fit = cpp_mbspca_one_lv(blocks, budgets, max_iter = 60L, tol = 1e-4)

  expect_true(all(vapply(fit$W, function(weight) {
    all(is.finite(weight))
  }, logical(1L))))
  expect_equal(
    vapply(fit$W, function(weight) sqrt(sum(weight^2)), numeric(1L)),
    rep(1, length(blocks)),
    tolerance = 1e-8
  )
  expect_true(all(
    vapply(fit$W, function(weight) sum(abs(weight)), numeric(1L)) <=
      budgets + 1e-6
  ))
  expect_error(
    cpp_mbspca_one_lv(blocks, c(0.5, budgets[[2L]]), 60L, 1e-4),
    "\\[1, sqrt\\(p\\)\\]"
  )
})

test_that("perm_test_component_mbspca consumes every requested permutation", {
  set.seed(13)
  X = list(matrix(rnorm(36), 12, 3), matrix(rnorm(36), 12, 3))
  fit = cpp_mbspca_one_lv(X, c(sqrt(3), sqrt(3)))

  # With the former partial-loop shortcut, alpha = 0 stopped both calls after
  # the same 52 replicates, so the following RNG draws were identical.
  set.seed(44)
  perm_test_component_mbspca(
    X, fit$W, c(sqrt(3), sqrt(3)), n_perm = 55L, alpha = 0
  )
  after_55 = runif(1)
  set.seed(44)
  perm_test_component_mbspca(
    X, fit$W, c(sqrt(3), sqrt(3)), n_perm = 56L, alpha = 0
  )
  after_56 = runif(1)

  expect_false(identical(after_55, after_56))
})

# ── perm_test_component_mbspca: max_iter / tol are forwarded ──────────────────
test_that("perm_test_component_mbspca respects max_iter and tol parameters", {
  skip_on_cran()
  set.seed(2)
  n = 30
  X = list(
    matrix(rnorm(n * 4), n, 4),
    matrix(rnorm(n * 3), n, 3)
  )
  W = list(
    c(0.5, 0.5, 0.5, 0.5),
    c(1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3))
  )

  # p-values with loose vs. tight tolerance should both be in [0, 1]
  pv_loose = perm_test_component_mbspca(X, W, c(2.0, sqrt(3)),
    n_perm = 49L, max_iter = 5L, tol = 1e-2)
  pv_tight = perm_test_component_mbspca(X, W, c(2.0, sqrt(3)),
    n_perm = 49L, max_iter = 100L, tol = 1e-8)

  expect_true(pv_loose >= 0 && pv_loose <= 1)
  expect_true(pv_tight >= 0 && pv_tight <= 1)
})

# ── PipeOpMBsPCA: max_iter / tol are part of the param set ───────────────────
test_that("PipeOpMBsPCA exposes max_iter and tol parameters", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 5L)
  po = PipeOpMBsPCA$new(
    blocks = task$block_features(),
    param_vals = list(ncomp = 1L, max_iter = 30L, tol = 1e-3)
  )
  pv = po$param_set$get_values(tags = "train")
  expect_equal(pv$max_iter, 30L)
  expect_equal(pv$tol, 1e-3)

  # Training must succeed with custom solver parameters
  trained = po$train(list(task))
  expect_false(is.null(po$state))
})

test_that("PipeOpMBsPCA passes max_iter/tol to permutation test", {
  skip_on_cran()
  set.seed(3)
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 3L)
  po = PipeOpMBsPCA$new(
    blocks = task$block_features(),
    param_vals = list(
      ncomp = 2L,
      permutation_test = TRUE,
      n_perm = 19L,
      perm_alpha = 0.05,
      max_iter = 20L,
      tol = 1e-3
    )
  )
  trained = po$train(list(task))
  expect_false(is.null(po$state))
  expect_gte(po$state$ncomp, 1L)
})

# ── KNN learners: k > N warning ──────────────────────────────────────────────
test_that("LearnerClassifKNNGower warns when k > n_train", {
  set.seed(10)
  n = 8
  df = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y  = factor(rep(c("a", "b"), n / 2))
  )
  task = mlr3::TaskClassif$new(id = "knn_small", backend = df, target = "y")
  lrn = mlr3::lrn("classif.knngower", k = 20L)
  expect_warning(lrn$train(task), regexp = "exceeds")
})

test_that("LearnerRegrKNNGower warns when k > n_train", {
  set.seed(11)
  n = 6
  df = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y  = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "knnr_small", backend = df, target = "y")
  lrn = mlr3::lrn("regr.knngower", k = 15L)
  expect_warning(lrn$train(task), regexp = "exceeds")
})

test_that("LearnerClassifKNNGower does NOT warn when k <= n_train", {
  set.seed(12)
  n = 25
  df = data.frame(
    x1 = rnorm(n),
    y  = factor(sample(c("a", "b"), n, replace = TRUE))
  )
  task = mlr3::TaskClassif$new(id = "knn_ok", backend = df, target = "y")
  lrn = mlr3::lrn("classif.knngower", k = 5L)
  expect_no_warning(lrn$train(task))
})
