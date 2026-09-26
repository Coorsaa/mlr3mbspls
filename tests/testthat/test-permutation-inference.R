test_that("complete-analysis permutation test refits and is reproducible", {
  x = matrix(seq(-2, 2, length.out = 30), ncol = 1L,
    dimnames = list(paste0("p", 1:30), "x"))
  blocks = list(x = x, y = x)
  analysis = function(current) {
    abs(stats::cor(current$x[, 1L], current$y[, 1L]))
  }

  set.seed(902)
  before_kind = RNGkind()
  before_seed = .Random.seed
  first = mb_permutation_test(
    blocks,
    analysis = analysis,
    permute_blocks = "y",
    n_perm = 49L,
    seed = 17L,
    analysis_seed = 18L
  )
  second = mb_permutation_test(
    blocks,
    analysis = analysis,
    permute_blocks = "y",
    n_perm = 49L,
    seed = 17L,
    analysis_seed = 18L
  )

  expect_s3_class(first, "mb_permutation_test")
  expect_identical(first$null_distribution, second$null_distribution)
  expect_equal(first$p_value, second$p_value)
  expect_gte(first$p_value, 1 / 50)
  expect_lte(first$p_value, 0.05)
  expect_identical(RNGkind(), before_kind)
  expect_identical(.Random.seed, before_seed)
  expect_identical(first$fixed_blocks, "x")
})

test_that("every alignment reruns the callback with a fixed fitting RNG", {
  blocks = list(x = matrix(1:8, ncol = 1L), y = matrix(1:8, ncol = 1L))
  state = new.env(parent = emptyenv())
  state$calls = list()
  result = mb_permutation_test(blocks, n_perm = 11L, analysis = function(current) {
    state$calls[[length(state$calls) + 1L]] = list(
      alignment = current$y[, 1L],
      random_start = stats::runif(3L)
    )
    stats::cor(current$x[, 1L], current$y[, 1L])
  })
  calls = state$calls
  expect_length(calls, 12L)
  expect_identical(calls[[1L]]$alignment, blocks$y[, 1L])
  expect_true(all(vapply(calls, function(call) {
    identical(call$random_start, calls[[1L]]$random_start)
  }, logical(1L))))
  expect_true(any(vapply(calls[-1L], function(call) {
    !identical(call$alignment, calls[[1L]]$alignment)
  }, logical(1L))))
  expect_equal(result$p_value, (result$exceedances + 1) / 12)
})

test_that("failed permutation analyses abort with context and restore RNG", {
  blocks = list(x = matrix(1:8, ncol = 1L), y = matrix(1:8, ncol = 1L))
  set.seed(410L)
  before = .Random.seed
  before_kind = RNGkind()
  state = new.env(parent = emptyenv())
  state$calls = 0L
  expect_error(mb_permutation_test(blocks, n_perm = 4L, analysis = function(current) {
    state$calls = state$calls + 1L
    stats::runif(1L)
    if (state$calls == 2L) stop("fit failure")
    1
  }), "Permutation 1 analysis failed: fit failure")
  expect_identical(state$calls, 2L)
  expect_identical(.Random.seed, before)
  expect_identical(RNGkind(), before_kind)
  expect_error(mb_permutation_test(blocks, analysis = function(current) NA_real_),
    "Observed analysis must return one finite")
})

test_that("permutation metadata uses the same numerical ties as its p-value", {
  blocks = list(x = matrix(1:8, ncol = 1L), y = matrix(1:8, ncol = 1L))
  result = mb_permutation_test(blocks, n_perm = 9L, analysis = function(current) {
    if (identical(current$x, current$y)) 1 else 1 - 10 * .Machine$double.eps
  })
  expect_equal(result$p_value, 1)
  expect_identical(result$exceedances, 9L)
  expect_equal(result$monte_carlo_tail_estimate, 1)
})

test_that("complete trajectories are the exchangeability unit", {
  unit = rep(paste0("p", 1:6), each = 2L)
  visit = rep(c("baseline", "followup"), 6L)
  block = matrix(seq_along(unit), ncol = 1L)
  blocks = list(x = block, y = block)
  analysis = function(current) {
    stats::cor(current$x[, 1L], current$y[, 1L])
  }

  result = mb_permutation_test(
    blocks,
    analysis = analysis,
    permute_blocks = "y",
    n_perm = 19L,
    exchangeability_unit = unit,
    within_unit = visit,
    seed = 31L
  )

  expect_identical(result$exchangeability_level, "whole_unit")
  expect_identical(result$n_exchangeability_units, 6L)
  expect_error(
    mb_permutation_test(
      blocks,
      analysis = analysis,
      permute_blocks = "y",
      n_perm = 2L,
      exchangeability_unit = unit
    ),
    "require `within_unit`"
  )

  incomplete_visit = visit
  incomplete_visit[length(incomplete_visit)] = "month12"
  expect_error(
    mb_permutation_test(
      blocks,
      analysis = analysis,
      permute_blocks = "y",
      n_perm = 2L,
      exchangeability_unit = unit,
      within_unit = incomplete_visit
    ),
    "same `within_unit` pattern"
  )
})

test_that("trajectory permutations preserve visits and strata despite unordered rows", {
  unit = c("a", "b", "a", "c", "d", "b", "c", "d", "e", "e")
  visit = c(1L, 2L, 2L, 1L, 2L, 1L, 2L, 1L, 1L, 2L)
  strata = ifelse(unit %in% c("a", "b"), "first",
    ifelse(unit %in% c("c", "d"), "second", "singleton"))
  block = cbind(unit = match(unit, unique(unit)), visit = visit,
    stratum = match(strata, unique(strata)))
  rownames(block) = paste0("row", seq_len(nrow(block)))
  state = new.env(parent = emptyenv())
  state$maps = list()
  mb_permutation_test(list(x = block, y = block), n_perm = 11L,
    exchangeability_unit = unit, within_unit = visit, strata = strata,
    analysis = function(current) {
      expect_identical(current$x, block)
      expect_identical(rownames(current$y), rownames(block))
      expect_identical(current$y[, "visit"], block[, "visit"])
      expect_identical(current$y[, "stratum"], block[, "stratum"])
      expect_identical(current$y[unit == "e", ], block[unit == "e", ])
      expect_true(all(vapply(split(current$y[, "unit"], unit), function(ids) {
        length(unique(ids)) == 1L
      }, logical(1L))))
      state$maps[[length(state$maps) + 1L]] = current$y[, "unit"]
      stats::cor(current$x[, "unit"], current$y[, "unit"])
    }
  )
  maps = state$maps
  expect_true(any(vapply(maps[-1L], function(map) !identical(map, maps[[1L]]),
    logical(1L))))
})

test_that("MB-sPLS wrapper returns one global omnibus p-value", {
  set.seed(44)
  n = 36L
  latent = stats::rnorm(n)
  blocks = list(
    clinical = cbind(a = latent + stats::rnorm(n, sd = 0.03),
      b = stats::rnorm(n)),
    imaging = cbind(c = latent + stats::rnorm(n, sd = 0.03),
      d = stats::rnorm(n))
  )

  result = mbspls_permutation_test(
    blocks,
    statistic = "global_lc1",
    n_perm = 49L,
    max_iter = 100L,
    seed = 8L,
    analysis_seed = 9L
  )

  expect_s3_class(result, "mbspls_permutation_test")
  expect_s3_class(result, "mb_permutation_test")
  expect_identical(result$statistic_name, "global_lc1")
  expect_length(result$component_statistics, 1L)
  expect_lte(result$p_value, 0.05)
  expect_match(result$validity_scope, "not rank-null p-values")
  expect_equal(unname(result$c_matrix[, 1L]), sqrt(c(2, 2)))
})


test_that("MB-sPLS permutation fitting is reproducible and RNG-local", {
  set.seed(45)
  blocks = list(
    first = matrix(stats::rnorm(48L), ncol = 2L),
    second = matrix(stats::rnorm(48L), ncol = 2L)
  )

  set.seed(46)
  before_kind = RNGkind()
  before_seed = .Random.seed
  first = mbspls_permutation_test(
    blocks,
    n_perm = 9L,
    max_iter = 40L,
    seed = 47L,
    analysis_seed = 48L
  )
  second = mbspls_permutation_test(
    blocks,
    n_perm = 9L,
    max_iter = 40L,
    seed = 47L,
    analysis_seed = 48L
  )

  expect_identical(first$null_distribution, second$null_distribution)
  expect_identical(first$statistic, second$statistic)
  expect_identical(RNGkind(), before_kind)
  expect_identical(.Random.seed, before_seed)
})

test_that("target statistic permutes only the named target block", {
  set.seed(78)
  n = 32L
  signal = stats::rnorm(n)
  blocks = list(
    x1 = cbind(a = signal + stats::rnorm(n, sd = 0.05), b = stats::rnorm(n)),
    x2 = cbind(c = signal + stats::rnorm(n, sd = 0.05), d = stats::rnorm(n)),
    target = cbind(y = signal + stats::rnorm(n, sd = 0.05), z = stats::rnorm(n))
  )

  result = mbspls_permutation_test(
    blocks,
    statistic = "target_lc1",
    target_block = "target",
    n_perm = 39L,
    max_iter = 100L,
    seed = 12L,
    analysis_seed = 13L
  )

  expect_identical(result$permute_blocks, "target")
  expect_identical(result$fixed_blocks, c("x1", "x2"))
  expect_match(result$null_hypothesis, "Target block")
  expect_lte(result$p_value, 0.05)
  expect_error(
    mbspls_permutation_test(
      blocks,
      statistic = "target_lc1",
      target_block = "target",
      permute_blocks = "x1",
      n_perm = 2L
    ),
    "target block only"
  )
})

test_that("MB-sPLS permutation wrapper rejects unsafe numeric inputs", {
  blocks = list(
    x = cbind(a = 1:6, constant = 1),
    y = cbind(b = 1:6, c = 6:1)
  )
  expect_error(
    mbspls_permutation_test(blocks, n_perm = 2L),
    "constant"
  )
  expect_error(
    mbspls_permutation_test(
      lapply(blocks, function(x) x[, 1L, drop = FALSE]),
      c_matrix = matrix(2, nrow = 2L),
      n_perm = 2L
    ),
    "sqrt\\(p_block\\)"
  )

  valid_blocks = list(
    x = cbind(a = 1:8, b = 8:1),
    y = cbind(c = c(1:7, 9), d = c(8:2, 0))
  )
  duplicated_rows = matrix(
    1.2,
    nrow = 3L,
    ncol = 1L,
    dimnames = list(c("x", "x", "y"), "LC1")
  )
  expect_error(
    mbspls_permutation_test(
      valid_blocks,
      c_matrix = duplicated_rows,
      n_perm = 2L
    ),
    "must have 2 rows"
  )

  duplicated_rows = matrix(
    1.2,
    nrow = 2L,
    ncol = 1L,
    dimnames = list(c("x", "x"), "LC1")
  )
  expect_error(
    mbspls_permutation_test(
      valid_blocks,
      c_matrix = duplicated_rows,
      n_perm = 2L
    ),
    "rows must be unique"
  )

  duplicated_components = matrix(
    1.2,
    nrow = 2L,
    ncol = 2L,
    dimnames = list(c("x", "y"), c("LC1", "LC1"))
  )
  expect_error(
    mbspls_permutation_test(
      valid_blocks,
      c_matrix = duplicated_components,
      n_perm = 2L
    ),
    "columns must be unique"
  )

  rank_limited = list(
    x = matrix(1:8, ncol = 1L),
    y = cbind(a = c(1:7, 9), b = c(8:2, 0))
  )
  expect_error(
    mbspls_permutation_test(rank_limited, ncomp = 2L, n_perm = 2L),
    "effective block rank"
  )
})

test_that("MB-sPLS checks integer component counts and rank after centering", {
  blocks = list(
    x = cbind(a = 1:8, b = (1:8) + 3),
    y = cbind(a = c(1:7, 9), b = c(8:2, 0))
  )
  expect_error(mbspls_permutation_test(blocks, ncomp = 2L, n_perm = 2L),
    "effective block rank")
  expect_error(mbspls_permutation_test(blocks, ncomp = 1.5,
    c_matrix = matrix(1.2, nrow = 2L), n_perm = 2L), "finite integer")
})

test_that("permutation blocks require explicit, unambiguous row alignment", {
  analysis = function(current) {
    stats::cor(current$x[, 1L], current$y[, 1L])
  }
  x = matrix(1:8, ncol = 1L, dimnames = list(paste0("p", 1:8), "x"))
  y = matrix(8:1, ncol = 1L)

  expect_error(
    mb_permutation_test(
      list(x = x, y = y),
      analysis = analysis,
      n_perm = 2L
    ),
    "every block or no block"
  )

  rownames(y) = c(paste0("p", 1:7), "p7")
  expect_error(
    mb_permutation_test(
      list(x = x, y = y),
      analysis = analysis,
      n_perm = 2L
    ),
    "row names must be unique"
  )

  empty = matrix(numeric(), nrow = 8L, ncol = 0L)
  expect_error(
    mb_permutation_test(
      list(x = unname(x), y = empty),
      analysis = analysis,
      n_perm = 2L
    ),
    "at least one column"
  )
})

test_that("frozen LCs can be tested on independent confirmation scores", {
  set.seed(401)
  n = 80L
  first = stats::rnorm(n)
  second = stats::rnorm(n)
  scores = list(
    imaging = cbind(
      LC1 = first + stats::rnorm(n, sd = 0.08),
      LC2 = second + stats::rnorm(n, sd = 0.08),
      LC3 = stats::rnorm(n)
    ),
    clinical = cbind(
      LC1 = first + stats::rnorm(n, sd = 0.08),
      LC2 = second + stats::rnorm(n, sd = 0.08),
      LC3 = stats::rnorm(n)
    )
  )

  set.seed(402)
  before_kind = RNGkind()
  before_seed = .Random.seed
  result = mb_lc_confirmation_test(
    scores,
    independent_confirmation = TRUE,
    permute_blocks = "clinical",
    n_perm = 99L,
    seed = 403L
  )

  expect_s3_class(result, "mb_lc_confirmation_test")
  expect_identical(result$adjustment, "holm")
  expect_identical(result$results$component, c("LC1", "LC2", "LC3"))
  expect_true(all(result$results$significant_holm[1:2]))
  expect_false(result$results$significant_holm[[3L]])
  expect_true(all(result$results$p_value_holm >= result$results$p_value_raw))
  expect_equal(result$minimum_raw_p_value, 0.01)
  expect_match(result$validity_scope, "Not a population-rank test")
  expect_identical(RNGkind(), before_kind)
  expect_identical(.Random.seed, before_seed)
})

test_that("LC confirmation inference fails closed on reused or malformed scores", {
  scores = list(
    a = cbind(LC1 = 1:8, LC2 = 8:1),
    b = cbind(LC1 = 1:8, LC2 = c(2:8, 1))
  )
  expect_error(
    mb_lc_confirmation_test(scores, n_perm = 9L),
    "independent_confirmation = TRUE"
  )

  mismatched = scores
  colnames(mismatched$b) = c("LC2", "LC1")
  expect_error(
    mb_lc_confirmation_test(
      mismatched,
      independent_confirmation = TRUE,
      n_perm = 9L
    ),
    "names/order differ"
  )

  constant = scores
  constant$b[, 1L] = 1
  expect_error(
    mb_lc_confirmation_test(
      constant,
      independent_confirmation = TRUE,
      n_perm = 9L
    ),
    "degenerate"
  )
})
