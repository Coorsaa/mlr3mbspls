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
  # Two strata of two units each admit only four distinct maps.
  expect_warning(mb_permutation_test(list(x = block, y = block), n_perm = 11L,
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
  ), class = "mb_small_permutation_group")
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

test_that("automatic data-frame row names are not treated as row IDs", {
  set.seed(301)
  x = matrix(stats::rnorm(20L), ncol = 2L)
  y = data.frame(a = stats::rnorm(10L), b = stats::rnorm(10L))
  analysis = function(current) {
    abs(stats::cor(current$x[, 1L], as.matrix(current$y)[, 1L]))
  }

  mixed = mb_permutation_test(list(x = x, y = y), analysis = analysis,
    n_perm = 9L, seed = 3L)
  matrices = mb_permutation_test(list(x = x, y = as.matrix(y)),
    analysis = analysis, n_perm = 9L, seed = 3L)
  expect_identical(mixed$null_distribution, matrices$null_distribution)
  expect_s3_class(
    mb_permutation_test(list(x = x, y = data.table::as.data.table(y)),
      analysis = analysis, n_perm = 9L),
    "mb_permutation_test"
  )

  fitted_mixed = mbspls_permutation_test(list(x = x, y = y), n_perm = 9L,
    seed = 4L)
  fitted_matrices = mbspls_permutation_test(list(x = x, y = as.matrix(y)),
    n_perm = 9L, seed = 4L)
  expect_identical(fitted_mixed$null_distribution,
    fitted_matrices$null_distribution)
  expect_identical(fitted_mixed$p_value, fitted_matrices$p_value)

  named = x
  rownames(named) = paste0("p", 1:10)
  expect_error(
    mb_permutation_test(list(x = named, y = y), analysis = analysis,
      n_perm = 2L),
    "every block or no block"
  )

  permuted = mlr3mbspls:::.mb_permute_block_rows(y, c(2:10, 1L))
  expect_lt(.row_names_info(permuted), 0L)
  expect_identical(permuted$a, y$a[c(2:10, 1L)])
  explicit = y
  rownames(explicit) = paste0("s", 1:10)
  expect_identical(
    rownames(mlr3mbspls:::.mb_permute_block_rows(explicit, c(2:10, 1L))),
    rownames(explicit)
  )
})

test_that("tibble blocks are permuted without row-name warnings", {
  skip_if_not_installed("tibble")
  set.seed(302)
  blocks = list(
    x = tibble::tibble(a = stats::rnorm(10L)),
    y = tibble::tibble(b = stats::rnorm(10L))
  )
  expect_no_warning(
    mb_permutation_test(blocks,
      analysis = function(current) stats::cor(current$x$a, current$y$b),
      n_perm = 5L)
  )
})

test_that("Monte Carlo precision is exact at the boundary", {
  x = matrix(seq(-2, 2, length.out = 30), ncol = 1L)
  result = mb_permutation_test(list(x = x, y = x), n_perm = 19L,
    analysis = function(current) {
      stats::cor(current$x[, 1L], current$y[, 1L])
    }
  )

  expect_identical(result$exceedances, 0L)
  expect_equal(result$p_value, 1 / 20)
  expect_equal(result$monte_carlo_tail_estimate, 0)
  expect_true(is.na(result$monte_carlo_standard_error))
  expect_equal(result$monte_carlo_conf_low, 0)
  expect_equal(result$monte_carlo_conf_high,
    stats::binom.test(0L, 19L)$conf.int[[2L]])
  expect_gt(result$monte_carlo_conf_high, 0)
  expect_identical(result$monte_carlo_conf_level, 0.95)
  expect_output(print(result),
    "Monte Carlo 95% CI for the exceedance probability: \\[0, 0.1765\\]")

  set.seed(303)
  noise = matrix(stats::rnorm(30L), ncol = 1L)
  interior = mb_permutation_test(list(x = x, y = noise), n_perm = 49L,
    analysis = function(current) {
      stats::cor(current$x[, 1L], current$y[, 1L])
    }, seed = 6L)
  tail = interior$exceedances / 49
  expect_gt(interior$exceedances, 0L)
  expect_lt(interior$exceedances, 49L)
  expect_equal(interior$monte_carlo_standard_error,
    sqrt(tail * (1 - tail) / 49))
})

test_that("LC confirmation reports exceedances and exact Monte Carlo intervals", {
  set.seed(304)
  n = 40L
  signal = stats::rnorm(n)
  scores = list(
    a = cbind(LC1 = signal + stats::rnorm(n, sd = 0.05),
      LC2 = stats::rnorm(n)),
    b = cbind(LC1 = signal + stats::rnorm(n, sd = 0.05),
      LC2 = stats::rnorm(n))
  )
  result = mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
    n_perm = 99L, seed = 7L)

  expect_true(all(c("exceedances", "monte_carlo_conf_low",
    "monte_carlo_conf_high") %in% names(result$results)))
  expect_identical(result$results$exceedances[[1L]], 0L)
  expect_equal(result$results$monte_carlo_conf_high[[1L]],
    stats::binom.test(0L, 99L)$conf.int[[2L]])
  expect_gt(result$results$monte_carlo_conf_high[[1L]], 0)
  expect_equal(result$results$p_value_raw,
    (result$results$exceedances + 1) / 100)
  expect_output(print(result), "Clopper-Pearson")
})

test_that("LC confirmation keeps its positional argument order", {
  set.seed(306)
  n = 30L
  signal = stats::rnorm(n)
  scores = list(
    a = cbind(LC1 = signal + stats::rnorm(n, sd = 0.2)),
    b = cbind(LC1 = signal + stats::rnorm(n, sd = 0.2))
  )
  positional = mb_lc_confirmation_test(scores, TRUE, "b", 19L)
  named = mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
    permute_blocks = "b", n_perm = 19L)
  expect_identical(positional$results, named$results)
  expect_identical(positional$tests[[1L]]$n_perm, 19L)
})

test_that("LC1 statistics refit one component per permutation", {
  set.seed(305)
  n = 40L
  latent = stats::rnorm(n)
  blocks = list(
    a = cbind(latent + stats::rnorm(n), stats::rnorm(n), stats::rnorm(n)),
    b = cbind(latent + stats::rnorm(n), stats::rnorm(n), stats::rnorm(n)),
    target = cbind(stats::rnorm(n), latent + stats::rnorm(n), stats::rnorm(n))
  )

  for (statistic in c("global_lc1", "target_lc1")) {
    target = if (statistic == "target_lc1") "target" else NULL
    one = mbspls_permutation_test(blocks, ncomp = 1L, statistic = statistic,
      target_block = target, n_perm = 9L, max_iter = 100L, seed = 3L)
    three = mbspls_permutation_test(blocks, ncomp = 3L, statistic = statistic,
      target_block = target, n_perm = 9L, max_iter = 100L, seed = 3L)
    expect_identical(three$statistic, one$statistic)
    expect_identical(three$null_distribution, one$null_distribution)
    expect_length(three$component_statistics, 3L)
    expect_length(three$component_objectives, 3L)
    expect_identical(three$component_statistics[[1L]], three$statistic)
    expect_identical(three$refit_ncomp, 1L)
    expect_identical(three$ncomp, 3L)
  }

  original = mlr3mbspls:::cpp_mbspls_multi_lv_cmatrix
  state = new.env(parent = emptyenv())
  local_mocked_bindings(
    cpp_mbspls_multi_lv_cmatrix = function(...) {
      state$columns = c(state$columns, ncol(list(...)$c_matrix))
      original(...)
    },
    .package = "mlr3mbspls"
  )
  state$columns = integer()
  mbspls_permutation_test(blocks, ncomp = 3L, statistic = "global_lc1",
    n_perm = 5L, max_iter = 100L)
  expect_identical(sort(state$columns), c(rep(1L, 6L), 3L))

  state$columns = integer()
  mbspls_permutation_test(blocks, ncomp = 3L, statistic = "global_sum",
    n_perm = 5L, max_iter = 100L)
  expect_identical(state$columns, rep(3L, 6L))
})

test_that("summed statistics use the component-major score layout", {
  scores_for = function(noise) {
    set.seed(306)
    n = 200L
    t1 = stats::rnorm(n)
    cbind(t1, t1 + noise * stats::rnorm(n), t1 + noise * stats::rnorm(n),
      stats::rnorm(n), stats::rnorm(n), stats::rnorm(n))
  }
  scores = scores_for(0.05)
  for (metric in c("mac", "frobenius")) {
    statistics = mlr3mbspls:::.mb_target_component_statistics(
      scores = scores, ncomp = 2L, block_names = c("x1", "x2", "target"),
      target_block = "target", correlation_method = "pearson",
      performance_metric = metric
    )
    expect_gt(statistics[[1L]], 0.95 * if (metric == "mac") 1 else sqrt(2))
    expect_lt(statistics[[2L]], 0.3)
  }

  set.seed(78)
  n = 40L
  first = stats::rnorm(n)
  second = stats::rnorm(n)
  blocks = list(
    x1 = cbind(first + stats::rnorm(n, sd = 0.3),
      second + stats::rnorm(n, sd = 0.5), stats::rnorm(n)),
    x2 = cbind(first + stats::rnorm(n, sd = 0.3),
      second + stats::rnorm(n, sd = 0.5), stats::rnorm(n)),
    target = cbind(first + stats::rnorm(n, sd = 0.3), stats::rnorm(n),
      second + stats::rnorm(n, sd = 0.5))
  )
  refit = function(c_matrix) {
    mlr3mbspls:::with_seed_local(21L, function() {
      mlr3mbspls:::cpp_mbspls_multi_lv_cmatrix(
        X_blocks = mlr3mbspls:::.mb_standardize_inference_blocks(blocks),
        c_matrix = c_matrix, max_iter = 100L, tol = 1e-4, spearman = FALSE,
        do_perm = FALSE, n_perm = 1L, alpha = 0.05, frobenius = FALSE
      )
    })
  }

  target_sum = mbspls_permutation_test(blocks, ncomp = 2L,
    statistic = "target_sum", target_block = "target", n_perm = 3L,
    max_iter = 100L, analysis_seed = 21L)
  fit = refit(target_sum$c_matrix)
  expected = vapply(1:2, function(component) {
    columns = (component - 1L) * 3L + 1:3
    # The solver objective independently identifies each component's columns.
    correlations = stats::cor(fit$T_mat[, columns])
    expect_equal(mean(abs(correlations[upper.tri(correlations)])),
      fit$objective[[component]], tolerance = 1e-8)
    mean(abs(correlations[3L, 1:2]))
  }, numeric(1L))
  expect_equal(target_sum$component_statistics, expected)
  expect_equal(target_sum$statistic, sum(expected))
  expect_identical(target_sum$refit_ncomp, 2L)

  global_sum = mbspls_permutation_test(blocks, ncomp = 2L,
    statistic = "global_sum", n_perm = 3L, max_iter = 100L,
    analysis_seed = 21L)
  fit = refit(global_sum$c_matrix)
  expect_equal(global_sum$component_statistics, as.numeric(fit$objective))
  expect_equal(global_sum$statistic, sum(fit$objective))
})

test_that("several permuted blocks receive independent maps", {
  n = 12L
  identity = matrix(seq_len(n), ncol = 1L)
  state = new.env(parent = emptyenv())
  state$maps = list()
  result = mb_permutation_test(
    list(a = identity, b = identity, c = identity),
    analysis = function(current) {
      state$maps[[length(state$maps) + 1L]] = current
      stats::cor(current$a[, 1L], current$b[, 1L])
    },
    n_perm = 20L,
    seed = 8L
  )

  maps = state$maps[-1L]
  expect_identical(result$permute_blocks, c("b", "c"))
  expect_true(all(vapply(maps, function(map) {
    identical(map$a, identity) &&
      identical(sort(map$b[, 1L]), seq_len(n)) &&
      identical(sort(map$c[, 1L]), seq_len(n))
  }, logical(1L))))
  expect_true(all(vapply(maps, function(map) {
    any(map$b[, 1L] != map$c[, 1L])
  }, logical(1L))))
})

test_that("the engine forwards two-sided and lower-tail alternatives", {
  x = matrix(seq(-2, 2, length.out = 20L), ncol = 1L)
  blocks = list(x = x, y = x[rev(seq_len(20L)), , drop = FALSE])
  analysis = function(current) stats::cor(current$x[, 1L], current$y[, 1L])

  two_sided = mb_permutation_test(blocks, analysis = analysis, n_perm = 29L,
    alternative = "two.sided", null_center = 0, seed = 9L)
  expect_equal(two_sided$statistic, -1)
  expect_identical(two_sided$null_center, 0)
  expect_equal(two_sided$p_value, mb_permutation_pvalue(two_sided$statistic,
    two_sided$null_distribution, "two.sided", null_center = 0))
  expect_identical(two_sided$exceedances,
    sum(abs(two_sided$null_distribution) >= 1 - 1e-12))

  less = mb_permutation_test(blocks, analysis = analysis, n_perm = 29L,
    alternative = "less", seed = 9L)
  expect_equal(less$p_value, mb_permutation_pvalue(less$statistic,
    less$null_distribution, "less"))
  expect_equal(less$p_value, 1 / 30)
  expect_true(is.na(less$null_center))

  greater = mb_permutation_test(blocks, analysis = analysis, n_perm = 29L,
    alternative = "greater", seed = 9L)
  expect_equal(greater$p_value, 1)

  expect_error(
    mb_permutation_test(blocks, analysis = analysis, n_perm = 2L,
      alternative = "two.sided"),
    "null_center"
  )
})

test_that("LC confirmation supports whole-unit and stratified designs", {
  set.seed(307)
  n_units = 12L
  unit = rep(paste0("u", seq_len(n_units)), each = 2L)
  visit = rep(c("baseline", "followup"), n_units)
  strata = rep(rep(c("site1", "site2"), each = n_units / 2L), each = 2L)
  signal = stats::rnorm(length(unit))
  scores = list(
    a = cbind(LC1 = signal + stats::rnorm(length(unit), sd = 0.1),
      LC2 = stats::rnorm(length(unit))),
    b = cbind(LC1 = signal + stats::rnorm(length(unit), sd = 0.1),
      LC2 = stats::rnorm(length(unit)))
  )

  result = mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
    n_perm = 39L, exchangeability_unit = unit, within_unit = visit,
    strata = strata, seed = 10L)
  expect_true(all(vapply(result$tests, `[[`, character(1L),
    "exchangeability_level") == "whole_unit"))
  expect_true(all(vapply(result$tests, `[[`, integer(1L),
    "n_exchangeability_units") == n_units))
  expect_equal(result$log_permutation_group_size, 2 * lfactorial(6))

  crossing = strata
  crossing[[1L]] = "site2"
  expect_error(
    mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
      n_perm = 9L, exchangeability_unit = unit, within_unit = visit,
      strata = crossing),
    "Strata vary within exchangeability units"
  )
})

test_that("small permutation groups are reported once per LC family", {
  unit = rep(paste0("site", 1:4), each = 5L)
  visit = rep(1:5, 4L)
  set.seed(308)
  scores = list(
    a = cbind(LC1 = stats::rnorm(20L), LC2 = stats::rnorm(20L)),
    b = cbind(LC1 = stats::rnorm(20L), LC2 = stats::rnorm(20L))
  )
  state = new.env(parent = emptyenv())
  state$caught = list()
  result = withCallingHandlers(
    mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
      n_perm = 49L, exchangeability_unit = unit, within_unit = visit),
    warning = function(condition) {
      state$caught[[length(state$caught) + 1L]] = condition
      invokeRestart("muffleWarning")
    }
  )
  classes = vapply(state$caught, function(condition) {
    inherits(condition, "mb_small_permutation_group")
  }, logical(1L))
  expect_identical(sum(classes), 1L)
  expect_equal(result$log_permutation_group_size, lfactorial(4))
})

test_that("an unreachable Holm threshold is flagged", {
  set.seed(309)
  scores = list(
    a = matrix(stats::rnorm(90L), ncol = 3L),
    b = matrix(stats::rnorm(90L), ncol = 3L)
  )
  expect_warning(
    mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
      n_perm = 19L),
    class = "mb_insufficient_permutations"
  )
  expect_no_warning(
    mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
      n_perm = 59L)
  )
})

test_that("LC confirmation direction is explicit and reported", {
  set.seed(310)
  n = 80L
  signal = stats::rnorm(n)
  reversed = list(
    predictor = cbind(LC1 = signal + stats::rnorm(n, sd = 0.3)),
    outcome = cbind(LC1 = -signal + stats::rnorm(n, sd = 0.3))
  )

  unsigned = mb_lc_confirmation_test(reversed, independent_confirmation = TRUE,
    n_perm = 99L, seed = 11L)
  expect_identical(unsigned$direction, "either")
  expect_true(unsigned$results$significant_holm[[1L]])
  expect_lt(unsigned$pairwise_correlations$correlation[[1L]], -0.5)
  expect_equal(unsigned$results$statistic[[1L]],
    abs(unsigned$pairwise_correlations$correlation[[1L]]))
  expect_match(unsigned$validity_scope, "Not a directional replication test")
  expect_match(unsigned$method, "either direction")
  expect_output(print(unsigned), "not a directional replication test")
  expect_output(print(unsigned), "Observed signed score correlations")

  same = mb_lc_confirmation_test(reversed, independent_confirmation = TRUE,
    n_perm = 99L, seed = 11L, reference_signs = c("predictor:outcome" = 1))
  expect_identical(same$direction, "expected_sign")
  expect_false(same$results$significant_holm[[1L]])
  expect_gt(same$results$p_value_raw[[1L]], 0.9)
  expect_lt(same$results$statistic[[1L]], 0)
  expect_match(same$validity_scope, "discovery direction")
  expect_output(print(same), "one-sided in the discovery direction")

  flipped = mb_lc_confirmation_test(reversed, independent_confirmation = TRUE,
    n_perm = 99L, seed = 11L, reference_signs = c("outcome:predictor" = -1))
  expect_true(flipped$results$significant_holm[[1L]])
  expect_equal(flipped$results$statistic[[1L]],
    unsigned$results$statistic[[1L]])
  expect_equal(flipped$pairwise_correlations$oriented_correlation[[1L]],
    -flipped$pairwise_correlations$correlation[[1L]])

  third = cbind(LC1 = signal + stats::rnorm(n, sd = 0.3))
  mixed_scores = c(reversed, list(imaging = third))
  signs = c("predictor:outcome" = -1, "predictor:imaging" = 1,
    "outcome:imaging" = -1)
  mixed = mb_lc_confirmation_test(mixed_scores,
    independent_confirmation = TRUE, n_perm = 49L, seed = 12L,
    reference_signs = signs)
  observed = stats::cor(do.call(cbind, lapply(mixed_scores, function(x) x[, 1L])))
  expect_equal(mixed$results$statistic[[1L]], mean(c(
    -observed["predictor", "outcome"], observed["predictor", "imaging"],
    -observed["outcome", "imaging"]
  )))
  expect_identical(mixed$pairwise_correlations$reference_sign, c(-1, 1, -1))
  expect_true(mixed$results$significant_holm[[1L]])

  per_lc = mb_lc_confirmation_test(
    list(a = cbind(L1 = signal, L2 = stats::rnorm(n)),
      b = cbind(L1 = -signal + stats::rnorm(n), L2 = stats::rnorm(n))),
    independent_confirmation = TRUE, n_perm = 59L,
    reference_signs = list(L2 = c("a:b" = 1), L1 = c("a:b" = -1))
  )
  expect_identical(per_lc$pairwise_correlations$reference_sign, c(-1, 1))
})

test_that("malformed reference signs fail with informative errors", {
  set.seed(311)
  scores = list(
    a = cbind(LC1 = stats::rnorm(20L)),
    b = cbind(LC1 = stats::rnorm(20L)),
    c = cbind(LC1 = stats::rnorm(20L))
  )
  run = function(reference_signs, ...) {
    mb_lc_confirmation_test(scores, independent_confirmation = TRUE,
      n_perm = 19L, reference_signs = reference_signs, ...)
  }
  expect_error(run(c("a:b" = 1, "a:c" = 1)), "missing: b:c")
  expect_error(run(c("a:b" = 1, "a:c" = 0, "b:c" = 1)), "must be -1 or 1")
  expect_error(run(c("a:b" = 1, "a:c" = NA, "b:c" = 1)), "must be -1 or 1")
  expect_error(run(c("a:b" = 1, "a:d" = 1, "b:c" = 1)), "Unknown block pairs")
  expect_error(run(c(1, 1, 1)), "named by block pair")
  expect_error(run(c("a:b" = 1, "b:a" = 1, "a:c" = 1, "b:c" = 1)),
    "only once")
  expect_error(run(list(c("a:b" = 1), c("a:b" = 1))), "one sign vector per LC")
  expect_error(run(c("a:b" = 1, "a:c" = 1, "b:c" = 1),
    performance_metric = "frobenius"), "performance_metric = \"mac\"")
  expect_error(run(list(LC9 = c("a:b" = 1, "a:c" = 1, "b:c" = 1))),
    "confirmation LC names")

  target_only = mb_lc_confirmation_test(scores,
    independent_confirmation = TRUE, permute_blocks = "c", n_perm = 19L,
    reference_signs = c("a:c" = 1, "b:c" = -1, "a:b" = 1))
  expect_identical(target_only$pairwise_correlations$block_2, c("c", "c"))
  expect_identical(target_only$pairwise_correlations$reference_sign, c(1, -1))
})

test_that("fixed-specification MB-sPLS tests carry their own label", {
  set.seed(312)
  n = 30L
  latent = stats::rnorm(n)
  blocks = list(
    clinical = cbind(a = latent + stats::rnorm(n, sd = 0.3),
      b = stats::rnorm(n)),
    imaging = cbind(c = latent + stats::rnorm(n, sd = 0.3),
      d = stats::rnorm(n))
  )
  result = mbspls_permutation_test(blocks, n_perm = 9L, max_iter = 100L,
    seed = 13L, analysis_seed = 14L)

  expect_match(result$method, "^Fixed-specification MB-sPLS omnibus")
  expect_match(result$validity_scope, "fixed independently of the tested alignment")
  expect_output(print(result), "Fixed-specification")
  expect_output(print(result), "Statistic: global_lc1; ncomp = 1")
  expect_output(print(result), "Scope: One omnibus p-value")
  expect_output(print(result), "Seeds: permutation = 13; analysis = 14")

  engine = mb_permutation_test(blocks, n_perm = 3L,
    analysis = function(current) {
      abs(stats::cor(current$clinical[, 1L], current$imaging[, 1L]))
    }
  )
  expect_output(print(engine), "Complete-analysis")
})

test_that("solver convergence is surfaced when the fit reports it", {
  set.seed(313)
  n = 30L
  latent = stats::rnorm(n)
  blocks = list(
    x = cbind(latent + stats::rnorm(n), stats::rnorm(n)),
    y = cbind(latent + stats::rnorm(n), stats::rnorm(n))
  )
  original = mlr3mbspls:::cpp_mbspls_multi_lv_cmatrix
  state = new.env(parent = emptyenv())
  local_mocked_bindings(
    cpp_mbspls_multi_lv_cmatrix = function(...) {
      state$calls = state$calls + 1L
      fit = original(...)
      # Absent or supplied convergence flags, whatever the solver reports.
      fit$converged = if (!is.null(state$converged)) {
        state$converged(state$calls, ncol(list(...)$c_matrix))
      }
      fit
    },
    .package = "mlr3mbspls"
  )

  state$calls = 0L
  state$converged = function(call, k) rep(!call %in% c(1L, 3L, 5L), k)
  expect_warning(
    {
      result = mbspls_permutation_test(blocks, n_perm = 9L, max_iter = 50L)
    },
    class = "mb_nonconverged_fit"
  )
  expect_identical(result$observed_converged, c(LC1 = FALSE))
  expect_identical(result$n_nonconverged_permutations, 2L)
  expect_output(print(result), "2 of 9 permutation refits did not converge")

  state$calls = 0L
  state$converged = function(call, k) rep(TRUE, k)
  expect_no_warning(
    {
      converged = mbspls_permutation_test(blocks, n_perm = 9L, max_iter = 50L)
    }
  )
  expect_identical(converged$n_nonconverged_permutations, 0L)
  expect_identical(converged$observed_converged, c(LC1 = TRUE))

  state$calls = 0L
  state$converged = NULL
  expect_no_warning(
    {
      unknown = mbspls_permutation_test(blocks, n_perm = 9L, max_iter = 50L)
    }
  )
  expect_identical(unknown$n_nonconverged_permutations, NA_integer_)
  expect_identical(unknown$observed_converged, c(LC1 = NA))
})
