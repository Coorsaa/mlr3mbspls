test_that("training scaler is frozen and schema-safe", {
  x = cbind(a = c(1, 2, 3), b = c(10, 20, 40))
  scaler = mb_fit_scaler(x)
  z = mb_apply_scaler(x[, c("b", "a")], scaler)

  expect_s3_class(scaler, "mbspls_scaler")
  expect_identical(colnames(z), c("a", "b"))
  expect_equal(colMeans(z), c(a = 0, b = 0), tolerance = 1e-12)
  expect_equal(unname(apply(z, 2L, stats::sd)), c(1, 1), tolerance = 1e-12)

  # New data are transformed with the training centre and scale.
  x_new = cbind(b = c(5, 100), a = c(7, -1))
  expected = sweep(
    sweep(x_new[, c("a", "b")], 2L, colMeans(x), "-"),
    2L, apply(x, 2L, stats::sd), "/"
  )
  expect_equal(mb_apply_scaler(x_new, scaler), expected)
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

  # Labels are shuffled across groups (not the identity) and the multiset of
  # group-level labels is preserved; row-level counts may change with group
  # sizes.
  first_of_group = function(v) v[match(unique(group), group)]
  outs = lapply(0:20, function(s) mb_permute_group_labels(y, group, seed = s))
  expect_true(any(!vapply(outs, identical, logical(1L), y)))
  for (out in outs) {
    expect_identical(
      sort(as.character(first_of_group(out))),
      sort(as.character(first_of_group(y)))
    )
  }
  two_groups = lapply(0:20, function(s) {
    mb_permute_group_labels(c("u", "u", "v"), c("g1", "g1", "g2"), seed = s)
  })
  expect_setequal(
    unique(vapply(two_groups, paste, character(1L), collapse = "")),
    c("uuv", "vvu")
  )
})

test_that("inconsistent groups are reported by name wherever they occur", {
  group = c("a", "b", "c", "a", "c", "d", "d")
  expect_error(
    mb_permute_group_labels(c(1, 2, 3, 1, 3, 4, 5), group, seed = 1),
    "vary within exchangeability groups: d\\.$"
  )
  expect_error(
    mb_cluster_bootstrap(group, strata = c("x", "x", "y", "x", "z", "y", "y")),
    "Strata vary within exchangeability groups: c\\.$"
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

  # Groups drawn more than once receive distinct bootstrap cluster ids.
  sizes = vapply(boot$sampled_groups, function(id) sum(group == id), integer(1L))
  expect_true(anyDuplicated(boot$sampled_groups) > 0L)
  expect_identical(length(unique(boot$bootstrap_cluster)), length(boot$sampled_groups))
  expect_identical(boot$bootstrap_cluster, rep(seq_along(boot$sampled_groups), unname(sizes)))
  expect_identical(group[boot$indices], rep(boot$sampled_groups, unname(sizes)))

  expect_error(
    mb_cluster_bootstrap(c("a", "a", "b"), c("x", "y", "x")),
    "vary within"
  )
})

test_that("cluster bootstrap draws match the per-group reference implementation", {
  reference_draw = function(group, strata, seed) {
    group_key = as.character(group)
    groups = unique(group_key)
    group_strata = if (!is.null(strata)) as.character(strata[match(groups, group_key)])
    draw = function() {
      if (is.null(group_strata)) {
        sampled = sample(groups, length(groups), replace = TRUE)
      } else {
        sampled = unlist(lapply(unique(group_strata), function(level) {
          eligible = groups[group_strata == level]
          sample(eligible, length(eligible), replace = TRUE)
        }), use.names = FALSE)
        sampled = sampled[sample.int(length(sampled), replace = FALSE)]
      }
      list(
        indices = unlist(lapply(sampled, function(id) which(group_key == id)), use.names = FALSE),
        sampled_groups = sampled,
        bootstrap_cluster = unlist(Map(function(id, draw_id) {
          rep.int(draw_id, sum(group_key == id))
        }, sampled, seq_along(sampled)), use.names = FALSE)
      )
    }
    getFromNamespace("with_seed_local", "mlr3mbspls")(seed, draw)
  }

  ids = c(3L, 1L, 2L, 3L, 5L, 1L, 4L, 4L, 2L, 5L, 6L, 3L)
  strata = c("x", "y", "x")[(ids %% 3L) + 1L]
  for (seed in c(1L, 7L, 42L)) {
    for (group in list(as.character(ids), factor(ids, levels = 6:1), ids)) {
      expect_identical(
        mb_cluster_bootstrap(group, seed = seed),
        reference_draw(group, NULL, seed)
      )
      expect_identical(
        mb_cluster_bootstrap(group, strata = strata, seed = seed),
        reference_draw(group, strata, seed)
      )
    }
  }
})

test_that("cluster bootstrap and group permutation scale linearly", {
  skip_on_cran()
  group = rep(seq_len(5000L), each = 4L)
  strata = group %% 3L
  elapsed = system.time({
    mb_cluster_bootstrap(group, strata = strata, seed = 1L)
    mb_permute_group_labels(group %% 2L, group, seed = 1L)
  })[["elapsed"]]
  expect_lt(elapsed, 2)
})

test_that("component signs are aligned to a reference", {
  reference = cbind(c1 = c(1, 2, 3), c2 = c(1, -1, 2))
  estimate = cbind(c1 = -reference[, 1], c2 = reference[, 2])
  aligned = mb_align_component_signs(estimate, reference)

  expect_equal(as.numeric(aligned), as.numeric(reference))
  expect_identical(dim(aligned), dim(reference))
  expect_equal(unname(attr(aligned, "signs")), c(-1, 1))
  expect_false(any(attr(aligned, "ambiguous")))
  expect_equal(attr(aligned, "cosines"), c(c1 = -1, c2 = 1))
})

test_that("sign ambiguity is judged by scale-free cosine similarity", {
  reference = cbind(c1 = c(1, 0, 0), c2 = c(0, 1, 0))
  near_orthogonal = c(-0.01, 1, 0) / sqrt(1 + 0.01^2)
  estimate = cbind(c1 = near_orthogonal, c2 = c(0.3, -2, 0.1))
  aligned = mb_align_component_signs(estimate, reference)

  expect_equal(
    attr(aligned, "cosines"),
    c(c1 = -0.01 / sqrt(1 + 0.01^2), c2 = -2 / sqrt(0.3^2 + 2^2 + 0.1^2))
  )
  expect_identical(attr(aligned, "ambiguous"), c(c1 = TRUE, c2 = FALSE))
  expect_identical(attr(aligned, "signs"), c(c1 = 1, c2 = -1))
  expect_equal(aligned[, "c1"], estimate[, "c1"])

  # Rescaling a column changes neither the flag nor the sign.
  for (scale in c(1e-6, 1e8)) {
    rescaled = mb_align_component_signs(estimate * scale, reference)
    expect_identical(attr(rescaled, "ambiguous"), attr(aligned, "ambiguous"))
    expect_identical(attr(rescaled, "signs"), attr(aligned, "signs"))
  }

  # Zero-norm columns are ambiguous.
  zero = mb_align_component_signs(cbind(c1 = c(0, 0, 0), c2 = c(0, -1, 0)), reference)
  expect_identical(attr(zero, "ambiguous"), c(c1 = TRUE, c2 = FALSE))
  expect_true(is.na(attr(zero, "cosines")[["c1"]]))

  # Swapped components are flagged rather than confidently aligned.
  swapped = cbind(c1 = c(0.05, 1, 0), c2 = c(-1, 0.02, 0))
  expect_true(all(attr(mb_align_component_signs(swapped, reference), "ambiguous")))

  expect_error(mb_align_component_signs(estimate, reference, tolerance = 1), "\\[0, 1\\)")
  expect_error(mb_align_component_signs(estimate, reference, tolerance = -0.1), "\\[0, 1\\)")
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

test_that("seeded evaluation restores an absent .Random.seed, also on error", {
  with_seed = getFromNamespace("with_seed_local", "mlr3mbspls")
  local({
    old_kind = RNGkind()
    old_seed = .Random.seed
    on.exit({
      do.call(RNGkind, as.list(old_kind))
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    })
    rm(".Random.seed", envir = .GlobalEnv)

    first = with_seed(3L, function() stats::runif(2L))
    expect_false(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    expect_error(with_seed(3L, function() stop("boom")), "boom")
    expect_false(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    expect_identical(with_seed(3L, function() stats::runif(2L)), first)
  })
})

test_that("seeded evaluation keeps seeded results, seed = NULL semantics, and the kind option", {
  with_seed = getFromNamespace("with_seed_local", "mlr3mbspls")
  local({
    old_kind = RNGkind()
    old_seed = .Random.seed
    on.exit({
      do.call(RNGkind, as.list(old_kind))
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    })
    draw = function() c(stats::runif(2L), stats::rnorm(2L), sample.int(10L, 2L))

    # The default generator reproduces set.seed() under R's default kinds.
    RNGkind("Mersenne-Twister", "Inversion", "Rejection")
    set.seed(11L)
    expect_identical(with_seed(11L, draw), draw())

    # seed = NULL evaluates unseeded in the caller's RNG stream.
    set.seed(5L)
    unseeded = with_seed(NULL, draw)
    set.seed(5L)
    expect_identical(unseeded, draw())
    expect_identical(with_seed(NULL, function() RNGkind()), RNGkind())

    lecuyer = with_seed(11L, draw, kind = "L'Ecuyer-CMRG")
    set.seed(11L, kind = "L'Ecuyer-CMRG", normal.kind = "Inversion", sample.kind = "Rejection")
    expect_identical(lecuyer, draw())
    RNGkind("Mersenne-Twister", "Inversion", "Rejection")
    expect_error(with_seed(1L, draw, kind = "Wichmann-Hill"), "should be one of")
  })
})

test_that("forked evaluation is reproducible with the L'Ecuyer-CMRG kind", {
  skip_on_os("windows")
  skip_on_cran()
  with_seed = getFromNamespace("with_seed_local", "mlr3mbspls")
  forked = function() {
    unlist(parallel::mclapply(1:2, function(i) stats::runif(1L), mc.cores = 2L))
  }
  expect_identical(
    with_seed(1L, forked, kind = "L'Ecuyer-CMRG"),
    with_seed(1L, forked, kind = "L'Ecuyer-CMRG")
  )
})

test_that("restoring a legacy sampler does not warn", {
  with_seed = getFromNamespace("with_seed_local", "mlr3mbspls")
  with_stream = getFromNamespace("with_rng_stream_local", "mlr3mbspls")
  local({
    old_kind = RNGkind()
    old_seed = .Random.seed
    on.exit({
      do.call(RNGkind, as.list(old_kind))
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    })
    suppressWarnings(RNGkind("Mersenne-Twister", "Inversion", "Rounding"))
    rounding = RNGkind()

    expect_no_warning(with_seed(1L, function() stats::runif(1L)))
    expect_identical(RNGkind(), rounding)
    expect_no_warning(mb_rng_streams(2L, 1L))
    expect_identical(RNGkind(), rounding)
    streams = mb_rng_streams(2L, 1L)
    expect_no_warning(with_stream(streams[[1L]], function() stats::runif(1L)))
    expect_identical(RNGkind(), rounding)
  })
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
    # Newer R versions report further kinds after the first three.
    expect_identical(RNGkind()[1:3], c("Wichmann-Hill", "Ahrens-Dieter", "Rejection"))
    expect_identical(.Random.seed, before)
    expect_error(with_stream(.Random.seed, function() runif(1L)), "L'Ecuyer")
    expect_error(with_stream(as.numeric(first[[1L]]), function() runif(1L)), "L'Ecuyer")
    # A Box-Muller normal generator keeps state outside the seed.
    box_muller = first[[1L]]
    box_muller[[1L]] = box_muller[[1L]] %/% 10000L * 10000L + 207L
    expect_error(with_stream(box_muller, function() runif(1L)), "normal generator")
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
  expect_identical(summary$interval_type, "percentile")
  expect_equal(summary$confidence_level, 0.8)
  expect_null(unclass(summary)[["conf"]])
  expect_null(unclass(summary)[["interval"]])
})

test_that("bootstrap intervals follow their closed forms with type-8 quantiles", {
  r = c(0.8, 0.9, 1.0, 1.1, 1.3)
  theta = 0.95
  conf = 0.8
  q = stats::quantile(r, c(0.1, 0.9), type = 8, names = FALSE)
  bias = mean(r) - theta
  endpoints = function(type) {
    s = mb_bootstrap_summary(r, theta, conf, type)
    c(s$conf_low, s$conf_high)
  }

  expect_equal(endpoints("percentile"), q)
  expect_equal(endpoints("basic"), c(2 * theta - q[[2L]], 2 * theta - q[[1L]]))
  expect_equal(
    endpoints("normal"),
    (theta - bias) + stats::qnorm(c(0.1, 0.9)) * stats::sd(r)
  )

  s = mb_bootstrap_summary(r, theta, conf, "basic")
  expect_equal(s$bootstrap_mean, mean(r))
  expect_equal(s$bias, bias)
  expect_equal(s$standard_error, stats::sd(r))
  expect_identical(s$interval_type, "basic")
  expect_identical(s$confidence_level, conf)
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

test_that("sign alignment is scale-free for extreme finite values", {
  ref = cbind(c1 = c(1e160, -1e160, 5e159), c2 = c(1e-300, 2e-300, -1e-300))
  est = cbind(c1 = -ref[, "c1"], c2 = 3 * ref[, "c2"])
  aligned = mb_align_component_signs(est, ref)
  expect_identical(unname(attr(aligned, "signs")), c(-1, 1))
  expect_false(any(attr(aligned, "ambiguous")))
  expect_equal(unname(attr(aligned, "cosines")), c(-1, 1))
  expect_equal(aligned[, "c1"], ref[, "c1"])
})
