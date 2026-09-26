# Name-based reference sampler. The production sampler precomputes integer
# structures and must reproduce these maps and the RNG stream exactly.
reference_exchangeability = function(n, exchangeability_unit = NULL,
  within_unit = NULL, strata = NULL) {
  strata_key = if (is.null(strata)) rep.int(".all", n) else as.character(strata)
  if (is.null(exchangeability_unit)) {
    return(list(n = n, strata = strata_key, unit_level = FALSE))
  }
  unit_key = as.character(exchangeability_unit)
  units = unique(unit_key)
  unit_rows = lapply(units, function(id) which(unit_key == id))
  names(unit_rows) = units
  within_key = if (is.null(within_unit)) {
    rep.int(".row", n)
  } else {
    as.character(within_unit)
  }
  list(
    n = n,
    within = within_key,
    units = units,
    unit_rows = unit_rows,
    unit_strata = vapply(unit_rows, function(rows) strata_key[rows[[1L]]],
      character(1L)),
    unit_level = TRUE
  )
}

reference_draw = function(exchangeability) {
  index = seq_len(exchangeability$n)
  if (!isTRUE(exchangeability$unit_level)) {
    for (level in unique(exchangeability$strata)) {
      rows = which(exchangeability$strata == level)
      index[rows] = rows[sample.int(length(rows), replace = FALSE)]
    }
    return(index)
  }
  for (level in unique(exchangeability$unit_strata)) {
    destination = exchangeability$units[exchangeability$unit_strata == level]
    source = destination[sample.int(length(destination), replace = FALSE)]
    for (i in seq_along(destination)) {
      destination_rows = exchangeability$unit_rows[[destination[[i]]]]
      source_rows = exchangeability$unit_rows[[source[[i]]]]
      position = match(
        exchangeability$within[destination_rows],
        exchangeability$within[source_rows]
      )
      index[destination_rows] = source_rows[position]
    }
  }
  index
}

expect_same_draws = function(n, ..., seeds = 1:15) {
  args = list(...)
  reference = do.call(reference_exchangeability, c(list(n = n), args))
  current = do.call(mlr3mbspls:::.mb_validate_exchangeability,
    c(list(n = n), args))
  for (seed in seeds) {
    set.seed(seed)
    expected = replicate(3L, reference_draw(reference))
    expected_state = .Random.seed
    set.seed(seed)
    actual = replicate(3L, mlr3mbspls:::.mb_draw_permutation_index(current))
    expect_identical(actual, expected)
    expect_identical(.Random.seed, expected_state)
  }
}

test_that("integer permutation sampler reproduces name-based maps exactly", {
  set.seed(2026)
  n_units = 24L
  unit = rep(paste0("s", seq_len(n_units)), each = 3L)
  visit = rep(c("screening", "baseline", "month6"), n_units)
  shuffled = sample(length(unit))
  unit = unit[shuffled]
  visit = visit[shuffled]
  unit_number = as.integer(sub("s", "", unit))
  strata = c("left", "middle", "right")[unit_number %% 3L + 1L]

  expect_same_draws(length(unit), exchangeability_unit = unit,
    within_unit = visit)
  expect_same_draws(length(unit), exchangeability_unit = unit,
    within_unit = visit, strata = strata)
  expect_same_draws(length(unit), exchangeability_unit = unit_number,
    within_unit = factor(visit, levels = c("month6", "baseline", "screening")),
    strata = factor(strata))

  lonely = strata
  lonely[unit == "s1"] = "alone"
  expect_same_draws(length(unit), exchangeability_unit = unit,
    within_unit = visit, strata = lonely)
  expect_same_draws(40L, exchangeability_unit = sample(40L))

  expect_same_draws(40L, strata = rep(seq_len(20L), each = 2L)[sample(40L)])
  expect_same_draws(41L, strata = c(rep(seq_len(20L), each = 2L), 21L))
  expect_same_draws(50L, strata = sample(letters[1:4], 50L, replace = TRUE))
  expect_same_draws(50L)
})

test_that("permutation maps are pinned for a fixed seed", {
  draw = function(n, ...) {
    exchangeability = mlr3mbspls:::.mb_validate_exchangeability(n, ...)
    set.seed(1L, kind = "Mersenne-Twister", normal.kind = "Inversion",
      sample.kind = "Rejection")
    mlr3mbspls:::.mb_draw_permutation_index(exchangeability)
  }
  old_kind = RNGkind()
  on.exit(do.call(RNGkind, as.list(old_kind)), add = TRUE)

  expect_identical(draw(10L), c(9L, 4L, 7L, 1L, 2L, 5L, 3L, 10L, 6L, 8L))
  expect_identical(
    draw(10L, strata = rep(c("a", "b"), c(4L, 6L))),
    c(1L, 3L, 4L, 2L, 9L, 7L, 6L, 8L, 5L, 10L)
  )
  expect_identical(
    draw(8L,
      exchangeability_unit = c("u3", "u1", "u2", "u4", "u1", "u3", "u4", "u2"),
      within_unit = c(2, 1, 1, 1, 2, 1, 2, 2)
    ),
    c(1L, 3L, 4L, 2L, 8L, 6L, 5L, 7L)
  )
})

test_that("permutation sampling scales linearly for cohort-sized designs", {
  skip_on_cran()
  n = 20000L
  elapsed = system.time({
    singleton = mlr3mbspls:::.mb_validate_exchangeability(
      n, exchangeability_unit = seq_len(n)
    )
    for (i in seq_len(10L)) mlr3mbspls:::.mb_draw_permutation_index(singleton)
  })[["elapsed"]]
  expect_lt(elapsed, 3)

  elapsed = system.time({
    pairs = mlr3mbspls:::.mb_validate_exchangeability(
      n, strata = rep(seq_len(n / 2L), each = 2L)
    )
    for (i in seq_len(10L)) mlr3mbspls:::.mb_draw_permutation_index(pairs)
  })[["elapsed"]]
  expect_lt(elapsed, 3)

  elapsed = system.time({
    visits = mlr3mbspls:::.mb_validate_exchangeability(
      n,
      exchangeability_unit = rep(seq_len(n / 2L), each = 2L),
      within_unit = rep(c("baseline", "followup"), n / 2L)
    )
    for (i in seq_len(10L)) mlr3mbspls:::.mb_draw_permutation_index(visits)
  })[["elapsed"]]
  expect_lt(elapsed, 3)
})

test_that("row-level strata are never crossed by a permutation", {
  strata = rep(c("s1", "s2", "s3"), c(5L, 7L, 8L))
  block = matrix(seq_along(strata), ncol = 1L)
  state = new.env(parent = emptyenv())
  state$maps = list()
  result = mb_permutation_test(
    list(x = block, y = block),
    analysis = function(current) {
      state$maps[[length(state$maps) + 1L]] = current$y[, 1L]
      stats::cor(current$x[, 1L], current$y[, 1L])
    },
    n_perm = 49L,
    strata = strata,
    seed = 5L
  )

  maps = state$maps[-1L]
  expect_true(all(vapply(maps, function(map) {
    identical(sort(map), seq_along(strata)) && all(strata[map] == strata)
  }, logical(1L))))
  expect_true(any(vapply(maps, function(map) any(map != seq_along(strata)),
    logical(1L))))
  expect_identical(result$exchangeability_level, "row")
  expect_identical(result$n_strata, 3L)
  expect_equal(result$log_permutation_group_size,
    lfactorial(5) + lfactorial(7) + lfactorial(8))

  expect_error(
    mb_permutation_test(list(x = block, y = block),
      analysis = function(current) 1, n_perm = 2L,
      strata = seq_along(strata)),
    "No stratum contains two exchangeable rows"
  )
})

test_that("design vector names must match block row names without reordering", {
  ids = paste0("id", 1:8)
  block = matrix(1:8, ncol = 1L, dimnames = list(ids, "value"))
  blocks = list(x = block, y = block)
  analysis = function(current) stats::cor(current$x[, 1L], current$y[, 1L])
  unit = stats::setNames(rep(paste0("u", 1:4), each = 2L), ids)
  visit = stats::setNames(rep(c("v1", "v2"), 4L), ids)
  strata = stats::setNames(rep(c("a", "b"), each = 4L), ids)
  run = function(...) {
    suppressWarnings(
      mb_permutation_test(blocks, analysis = analysis, n_perm = 3L, ...),
      classes = "mb_small_permutation_group"
    )
  }
  shuffled = rev(ids)

  expect_error(
    run(exchangeability_unit = unit[shuffled], within_unit = visit),
    "Names of `exchangeability_unit` do not match the block row names"
  )
  expect_error(
    run(exchangeability_unit = unit, within_unit = visit[shuffled]),
    "Names of `within_unit` do not match the block row names"
  )
  expect_error(
    run(strata = strata[shuffled]),
    "Names of `strata` do not match the block row names"
  )

  reordered = unit[shuffled][rownames(block)]
  expect_s3_class(
    run(exchangeability_unit = reordered, within_unit = visit, strata = strata),
    "mb_permutation_test"
  )
  expect_s3_class(
    run(exchangeability_unit = unname(unit[shuffled]),
      within_unit = unname(visit[shuffled])),
    "mb_permutation_test"
  )

  frames = list(
    x = data.frame(value = 1:8),
    y = data.frame(value = 1:8)
  )
  expect_s3_class(
    mb_permutation_test(frames,
      analysis = function(current) stats::cor(current$x$value, current$y$value),
      n_perm = 3L, strata = stats::setNames(strata, paste0("sample", 1:8))
    ),
    "mb_permutation_test"
  )
})

test_that("permutation group size reflects units, strata, and permuted blocks", {
  unit = rep(paste0("site", 1:4), each = 5L)
  visit = rep(1:5, 4L)
  block = matrix(seq_along(unit), ncol = 1L)
  analysis = function(current) stats::cor(current$x[, 1L], current$y[, 1L])

  expect_warning(
    {
      result = mb_permutation_test(list(x = block, y = block),
        analysis = analysis, n_perm = 99L,
        exchangeability_unit = unit, within_unit = visit)
    },
    class = "mb_small_permutation_group"
  )
  expect_equal(result$log_permutation_group_size, lfactorial(4))
  expect_equal(result$permutation_group_size, 24)
  expect_equal(result$minimum_p_value, 0.01)
  expect_output(print(result), "Permutation group size: 24 distinct maps")

  expect_warning(
    {
      three = mb_permutation_test(list(x = block, y = block, z = block),
        analysis = analysis, n_perm = 999L,
        exchangeability_unit = unit, within_unit = visit)
    },
    class = "mb_small_permutation_group"
  )
  expect_equal(three$log_permutation_group_size, 2 * lfactorial(4))

  stratified = mb_permutation_test(list(x = block, y = block),
    analysis = analysis, n_perm = 3L,
    exchangeability_unit = unit, within_unit = visit,
    strata = rep(c("a", "b"), c(10L, 10L)))
  expect_equal(stratified$log_permutation_group_size, 2 * lfactorial(2))

  mixed_unit = rep(paste0("u", 1:5), each = 2L)
  mixed_block = matrix(seq_along(mixed_unit), ncol = 1L)
  mixed = mb_permutation_test(list(x = mixed_block, y = mixed_block),
    analysis = analysis, n_perm = 3L, exchangeability_unit = mixed_unit,
    within_unit = rep(1:2, 5L), strata = rep(c("a", "b"), c(4L, 6L)))
  expect_equal(mixed$log_permutation_group_size, lfactorial(2) + lfactorial(3))

  rows = matrix(stats::rnorm(30L), ncol = 1L)
  expect_no_warning(
    {
      row_level = mb_permutation_test(list(x = rows, y = rows),
        analysis = analysis, n_perm = 99L)
    }
  )
  expect_equal(row_level$log_permutation_group_size, lfactorial(30))
})
