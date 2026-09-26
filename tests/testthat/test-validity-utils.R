test_that("training scaler is frozen and schema-safe", {
  x = cbind(a = c(1, 2, 3), b = c(10, 20, 40))
  scaler = mb_fit_scaler(x)
  z = mb_apply_scaler(x[, c("b", "a")], scaler)

  expect_s3_class(scaler, "mbspls_scaler")
  expect_identical(colnames(z), c("a", "b"))
  expect_equal(colMeans(z), c(a = 0, b = 0), tolerance = 1e-12)
  expect_error(
    mb_apply_scaler(cbind(a = 1:3, c = 1:3), scaler),
    "schema"
  )
  expect_error(mb_fit_scaler(cbind(a = 1:3, b = 1)), "Zero-variance")
  expect_error(
    mb_fit_scaler(cbind(a = c(1, NA, 3), b = 1:3)),
    "finite"
  )

  bad_names = scaler
  names(bad_names$scale) = c("a", "wrong")
  expect_error(mb_apply_scaler(x, bad_names), "invalid fitted")

  bad_length = scaler
  bad_length$center = bad_length$center[[1L]]
  expect_error(mb_apply_scaler(x, bad_length), "invalid fitted")

  bad_scale = scaler
  bad_scale$scale[[1L]] = 0
  expect_error(mb_apply_scaler(x, bad_scale), "invalid fitted")
})

test_that("sampled permutation p-values use inclusive ranks and correction", {
  expect_equal(
    mb_permutation_pvalue(5, c(1, 2, 3), "greater"),
    1 / 4
  )
  expect_equal(
    mb_permutation_pvalue(2, c(2, 3, 4), "greater"),
    1
  )
  expect_equal(
    mb_permutation_pvalue(
      -2, c(-3, -2, 0, 2), "two.sided", null_center = 0
    ),
    4 / 5
  )
  expect_error(
    mb_permutation_pvalue(-2, c(-3, 0, 2), "two.sided"),
    "null_center"
  )
  expect_error(
    mb_permutation_pvalue(1, c(1, NA), "greater"),
    "finite"
  )
})

test_that("numerical permutation ties are counted in every tail", {
  near_one = 1 - 10 * .Machine$double.eps
  expect_equal(mb_permutation_pvalue(1, c(0, near_one)), 2 / 3)
  expect_equal(mb_permutation_pvalue(-1, c(0, -near_one), "less"), 2 / 3)
  expect_equal(
    mb_permutation_pvalue(-1, c(0, near_one), "two.sided", null_center = 0),
    2 / 3
  )
  expect_equal(mb_permutation_pvalue(1, c(0, 1 - 1e-8)), 1 / 3)
  expect_error(mb_permutation_pvalue(1, matrix(1:4, 2L)), "vector")
})

test_that("group-label permutation respects exchangeability units", {
  y = factor(c("case", "case", "control", "control", "case"))
  group = c("p1", "p1", "p2", "p2", "p3")
  out1 = mb_permute_group_labels(y, group, seed = 19)
  out2 = mb_permute_group_labels(y, group, seed = 19)

  expect_identical(out1, out2)
  expect_identical(levels(out1), levels(y))
  expect_true(all(vapply(split(out1, group), function(value) {
    length(unique(value)) == 1L
  }, logical(1L))))
  expect_error(
    mb_permute_group_labels(c(0, 1, 0), c("a", "a", "b"), seed = 1),
    "vary within"
  )
})

test_that("group-label permutation preserves destination names and attributes", {
  labels = factor(c("case", "case", "control", "control"), ordered = TRUE)
  names(labels) = paste0("row", seq_along(labels))
  attr(labels, "label") = "diagnosis"
  permuted = mb_permute_group_labels(labels, c("a", "a", "b", "b"), seed = 1L)

  expect_identical(attributes(permuted), attributes(labels))
  expect_identical(sort(as.character(permuted)), sort(as.character(labels)))
})

test_that("cluster bootstrap keeps clusters intact and respects strata", {
  group = c("a", "a", "b", "c", "c", "c", "d")
  strata = c("x", "x", "x", "y", "y", "y", "y")
  boot = mb_cluster_bootstrap(group, strata = strata, seed = 42)

  expect_length(boot$sampled_groups, length(unique(group)))
  expect_length(boot$indices, sum(vapply(
    boot$sampled_groups,
    function(id) sum(group == id),
    numeric(1L)
  )))
  expect_equal(
    table(strata[match(boot$sampled_groups, group)]),
    table(strata[match(unique(group), group)])
  )
  for (id in unique(boot$bootstrap_cluster)) {
    rows = boot$indices[boot$bootstrap_cluster == id]
    expect_length(unique(group[rows]), 1L)
  }
  expect_error(
    mb_cluster_bootstrap(c("a", "a", "b"), c("x", "y", "x")),
    "vary within"
  )
})

test_that("component signs are aligned to a reference", {
  reference = cbind(c1 = c(1, 2, 3), c2 = c(1, -1, 2))
  estimate = cbind(c1 = -reference[, 1], c2 = reference[, 2])
  aligned = mb_align_component_signs(estimate, reference)

  expect_equal(as.numeric(aligned), as.numeric(reference))
  expect_identical(dim(aligned), dim(reference))
  expect_equal(unname(attr(aligned, "signs")), c(-1, 1))
  expect_false(any(attr(aligned, "ambiguous")))
})

test_that("seeded evaluation and streams preserve caller RNG context", {
  with_seed = getFromNamespace("with_seed_local", "mlr3mbspls")
  with_stream = getFromNamespace("with_rng_stream_local", "mlr3mbspls")

  set.seed(101)
  before_kind = RNGkind()
  before_seed = .Random.seed
  first = with_seed(0, function() stats::runif(3))
  expect_identical(RNGkind(), before_kind)
  expect_identical(.Random.seed, before_seed)
  second = with_seed(0, function() stats::runif(3))
  expect_identical(first, second)

  streams = mb_rng_streams(4, 99)
  expect_identical(RNGkind(), before_kind)
  expect_identical(.Random.seed, before_seed)
  expect_length(unique(vapply(streams, function(stream) {
    paste(stream, collapse = ":")
  }, character(1L))), 4L)

  stream_draws = lapply(streams, function(stream) {
    with_stream(stream, function() stats::runif(2))
  })
  expect_identical(.Random.seed, before_seed)
  expect_length(unique(vapply(stream_draws, paste,
    FUN.VALUE = character(1L), collapse = ":")), 4L)
})

test_that("RNG streams are independent of ambient generators and reject invalid states", {
  with_stream = getFromNamespace("with_rng_stream_local", "mlr3mbspls")
  local({
    old_kind = RNGkind()
    old_seed = .Random.seed
    on.exit({
      do.call(RNGkind, as.list(old_kind))
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    })
    first = mb_rng_streams(2L, 42L)
    RNGkind("Wichmann-Hill", "Ahrens-Dieter", "Rejection")
    before = .Random.seed
    second = mb_rng_streams(2L, 42L)
    expect_identical(first, second)
    expect_identical(RNGkind(), c("Wichmann-Hill", "Ahrens-Dieter", "Rejection"))
    expect_identical(.Random.seed, before)
    expect_error(with_stream(.Random.seed, function() runif(1L)), "L'Ecuyer")
    expect_error(with_stream(as.numeric(first[[1L]]), function() runif(1L)), "L'Ecuyer")
    bad = first[[1L]]
    bad[2:4] = 0L
    expect_error(with_stream(bad, function() runif(1L)), "generator state")
    bad = first[[1L]]
    bad[[2L]] = -1L
    expect_error(with_stream(bad, function() runif(1L)), "generator state")

    expect_error(with_stream(first[[1L]], function() stop("failed draw")), "failed draw")
    expect_identical(.Random.seed, before)
    draw = function() c(runif(2L), rnorm(2L), sample.int(10L, 2L))
    expect_identical(
      with_stream(first[[1L]], draw),
      with_stream(first[[1L]], draw)
    )
    unsigned_state = first[[1L]]
    unsigned_state[[2L]] = NA_integer_
    expect_true(all(is.finite(with_stream(unsigned_state, draw))))

    rm(".Random.seed", envir = .GlobalEnv)
    streams = mb_rng_streams(2L, 42L)
    expect_false(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    expect_length(with_stream(streams[[1L]], draw), 6L)
    expect_false(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    expect_error(with_stream(streams[[1L]], function() stop("failed draw")), "failed draw")
    expect_false(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
  })
})

test_that("bootstrap output is descriptive uncertainty, not a null test", {
  summary = mb_bootstrap_summary(
    c(0.8, 0.9, 1.0, 1.1, 1.2, NA),
    observed = 1,
    conf = 0.8,
    type = "percentile"
  )

  expect_s3_class(summary, "mbspls_bootstrap_summary")
  expect_true(is.na(summary$p_value))
  expect_match(summary$p_value_note, "not a null distribution",
    ignore.case = TRUE)
  expect_equal(summary$bias, 0, tolerance = 1e-12)
  expect_lt(summary$conf_low, summary$conf_high)
  expect_identical(summary$replicates_requested, 6L)
  expect_identical(summary$replicates_effective, 5L)
  expect_identical(summary$replicates_failed, 1L)
})

test_that("group leakage across resampling partitions is rejected", {
  expect_invisible(
    mb_assert_disjoint_groups(c("p1", "p2"), c("p3", "p4"))
  )
  expect_error(
    mb_assert_disjoint_groups(c("p1", "p2"), c("p2", "p3")),
    "leakage"
  )
})
