test_that("PipeOpSiteCorrection - dir method errors when protected attribute has only one level", {
  set.seed(7)
  n = 40
  df = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    # Protected attribute is constant (only one level) — must fail
    prot = factor(rep("p0", n)),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "mb_dir_onelevel", backend = df, target = "y")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = list(b1 = c("x1", "x2")),
      site_correction = list(b1 = "prot"),
      method = list(b1 = "dir"),
      keep_site_col = FALSE
    )
  )

  expect_error(
    po$train(list(task)),
    "at least 2 levels"
  )
})


test_that("PipeOpSiteCorrection - dir method succeeds when protected attribute has 2+ levels", {
  set.seed(9)
  n = 40
  df = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    prot = factor(sample(c("p0", "p1"), n, replace = TRUE)),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "mb_dir_twolevels", backend = df, target = "y")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = list(b1 = c("x1", "x2")),
      site_correction = list(b1 = "prot"),
      method = list(b1 = "dir"),
      keep_site_col = FALSE
    )
  )

  out = po$train(list(task))[[1]]
  expect_s3_class(out, "Task")
  expect_equal(out$nrow, n)
})


test_that("PipeOpSiteCorrection - combat predict emits warning when covariates were used at training", {
  testthat::skip_if_not_installed("neuroCombat")

  set.seed(2)
  n = 60
  df_train = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    site = factor(sample(c("A", "B", "C"), n, replace = TRUE)),
    age = rnorm(n, 50, 10),
    y = rnorm(n)
  )
  task_train = mlr3::TaskRegr$new(id = "mb_combat_cov_train", backend = df_train, target = "y")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = list(b1 = c("x1", "x2")),
      # Combat with covariates at train time
      site_correction = list(b1 = list(site = "site", covariates = "age")),
      method = list(b1 = "combat"),
      keep_site_col = TRUE
    )
  )

  po$train(list(task_train))

  # Predict should emit a warning about covariates not being re-applied
  expect_warning(
    po$predict(list(task_train)),
    "covariate"
  )
})


test_that("PipeOpSiteCorrection - combat predict without covariates emits no extra warning", {
  testthat::skip_if_not_installed("neuroCombat")

  set.seed(3)
  n = 60
  df_train = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    site = factor(sample(c("A", "B"), n, replace = TRUE)),
    y = rnorm(n)
  )
  task_train = mlr3::TaskRegr$new(id = "mb_combat_no_cov", backend = df_train, target = "y")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = list(b1 = c("x1", "x2")),
      site_correction = list(b1 = list(site = "site", covariates = character(0))),
      method = list(b1 = "combat"),
      keep_site_col = TRUE
    )
  )

  po$train(list(task_train))

  # No covariate warning expected
  expect_no_warning(
    po$predict(list(task_train))
  )
})


test_that("PipeOpSiteCorrection validates maps, subgroups, and finite features", {
  set.seed(4)
  n = 20L
  task = mlr3::TaskRegr$new(
    "sitecorr_validation",
    data.frame(
      x1 = rnorm(n),
      x2 = rnorm(n),
      site = factor(rep(c("A", "B"), each = n / 2L)),
      y = rnorm(n)
    ),
    target = "y"
  )
  blocks = list(b1 = c("x1", "x2"))

  empty_maps = PipeOpSiteCorrection$new(param_vals = list(
    blocks = blocks,
    site_correction = list(),
    method = list()
  ))
  expect_s3_class(empty_maps$train(list(task))[[1L]], "Task")

  unknown_map = PipeOpSiteCorrection$new(param_vals = list(
    blocks = blocks,
    site_correction = list(wrong = "site")
  ))
  expect_error(unknown_map$train(list(task)), "unknown or unusable blocks")

  invalid_method = PipeOpSiteCorrection$new(param_vals = list(
    blocks = blocks,
    site_correction = list(b1 = "site"),
    method = list(b1 = "unknown")
  ))
  expect_error(invalid_method$train(list(task)), "Invalid site-correction")

  bad_subgroup = PipeOpSiteCorrection$new(param_vals = list(
    blocks = blocks,
    site_correction = list(b1 = "site"),
    subgroup = c(TRUE, FALSE)
  ))
  expect_error(bad_subgroup$train(list(task)), "one entry per training row")

  one_row = PipeOpSiteCorrection$new(param_vals = list(
    blocks = blocks,
    site_correction = list(b1 = "site"),
    subgroup = 1L
  ))
  expect_error(one_row$train(list(task)), "at least two training rows")

  missing_data = task$data()
  missing_data$x1[[1L]] = NA_real_
  missing_task = mlr3::TaskRegr$new(
    "sitecorr_missing",
    missing_data,
    target = "y"
  )
  missing_pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = blocks,
    site_correction = list(b1 = "site")
  ))
  expect_error(missing_pipeop$train(list(missing_task)), "finite")
})


test_that("PipeOpSiteCorrection freezes multivariable factor schemas", {
  set.seed(5)
  n = 30L
  train_task = mlr3::TaskRegr$new(
    "sitecorr_schema_train",
    data.frame(
      x1 = rnorm(n),
      x2 = rnorm(n),
      site = factor(rep(c("A", "B"), each = n / 2L)),
      age = rnorm(n, 50, 8),
      y = rnorm(n)
    ),
    target = "y"
  )
  pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = c("x1", "x2")),
    site_correction = list(b1 = c("site", "age")),
    method = list(b1 = "partial_corr"),
    keep_site_col = TRUE
  ))
  pipeop$train(list(train_task))

  prediction_task = mlr3::TaskRegr$new(
    "sitecorr_schema_predict",
    data.frame(
      x1 = rnorm(4L),
      x2 = rnorm(4L),
      site = factor(c("A", "C", "A", "C")),
      age = rnorm(4L, 50, 8),
      y = rnorm(4L)
    ),
    target = "y"
  )
  expect_error(
    pipeop$predict(list(prediction_task)),
    "contains unseen levels: C"
  )
  # `unknown_site` only applies to single categorical site specifications.
  for (strategy in c("other", "baseline")) {
    strict = pipeop$clone(deep = TRUE)
    strict$param_set$set_values(unknown_site = strategy)
    expect_error(
      strict$predict(list(prediction_task)),
      "contains unseen levels: C.*single categorical site column"
    )
  }

  corrupted = pipeop$clone(deep = TRUE)
  names(corrupted$state$per_block$b1$means) = c("x1", "wrong")
  expect_error(
    corrupted$predict(list(train_task)),
    "missing 1 trained feature name"
  )
})


test_that("unseen categorical sites are unchanged and coding is frozen", {
  train_task = mlr3::TaskRegr$new("site_noop_train", data.frame(
    x = c(1, 3, 11, 13),
    site = factor(c(".other", ".other", "B", "B")),
    y = 1:4
  ), target = "y")
  pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site")
  ))
  trained = pipeop$train(list(train_task))[[1L]]

  old_options = options(contrasts = c("contr.sum", "contr.poly"))
  on.exit(options(old_options), add = TRUE)
  predicted = pipeop$predict(list(train_task))[[1L]]
  expect_equal(predicted$data(cols = "x"), trained$data(cols = "x"))

  new_task = mlr3::TaskRegr$new("site_noop_predict", data.frame(
    x = c(100, 200), site = factor(c("new", NA)), y = 1:2
  ), target = "y")
  predicted = pipeop$predict(list(new_task))[[1L]]
  expect_equal(predicted$data(cols = "x")$x, c(100, 200))
})


test_that("DIR reuses training repair maps for single rows and batches", {
  train_task = mlr3::TaskRegr$new("dir_map_train", data.frame(
    x = c(1, 2, 4, 8, 11, 12, 14, 18),
    site = factor(rep(c("A", "B"), each = 4)), y = 1:8
  ), target = "y")
  pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
    method = list(b1 = "dir"), lambda = 0.5
  ))
  trained = pipeop$train(list(train_task))[[1L]]
  replayed = pipeop$predict(list(train_task))[[1L]]
  expect_equal(replayed$data(cols = "x"), trained$data(cols = "x"))

  new_task = mlr3::TaskRegr$new("dir_map_predict", data.frame(
    x = c(3, 1000, -1000), site = factor(c("A", "B", "A")), y = 1:3
  ), target = "y")
  batch = pipeop$predict(list(new_task))[[1L]]
  single = new_task$clone(deep = TRUE)$filter(1L)
  one = pipeop$predict(list(single))[[1L]]
  expect_equal(one$data(cols = "x")$x, batch$data(cols = "x")$x[[1L]])
  expect_true(all(is.finite(batch$data(cols = "x")$x)))

  zero = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
    method = list(b1 = "dir"), lambda = 0
  ))
  expect_equal(zero$train(list(train_task))[[1L]]$data(cols = "x"),
    train_task$data(cols = "x"))
  expect_equal(zero$predict(list(new_task))[[1L]]$data(cols = "x"),
    new_task$data(cols = "x"))
})


test_that("unpenalized site regression rejects unidentifiable designs", {
  task = mlr3::TaskRegr$new("site_singular", data.frame(
    x = c(3, 5, 8, 11, 14, 20),
    age = 1:6, age_duplicate = 2 * (1:6),
    site = factor(rep(c("A", "B"), each = 3)), y = 1:6
  ), target = "y")
  redundant = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"),
    site_correction = list(b1 = c("age", "age_duplicate"))
  ))
  expect_error(redundant$train(list(task)), "rank deficient")

  omitted_site = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
    subgroup = 1:3
  ))
  expect_error(omitted_site$train(list(task)), "rank deficient")

  regularized = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"),
    site_correction = list(b1 = c("age", "age_duplicate")),
    regularization = 1
  ))
  expect_true(all(is.finite(
    regularized$train(list(task))[[1L]]$data(cols = "x")$x
  )))
})


test_that("native ridge honors one-based unpenalized column indices", {
  design = cbind(1, seq_len(8), c(3, 1, 4, 2, 5, 8, 7, 6))
  response = matrix(c(3, 5, 8, 11, 14, 20, 23, 30), ncol = 1)
  penalty = diag(c(0, 0, 2))
  expected = solve(crossprod(design) + penalty, crossprod(design, response))
  coefficients = mlr3mbspls:::cpp_lm_coeff_ridge(design, response, 2, c(1L, 2L))
  expect_equal(unname(coefficients), unname(expected), tolerance = 1e-10)
})


test_that("site correction preserves targets colliding with internal key names", {
  task = mlr3::TaskRegr$new("site_key_target", data.frame(
    x = c(1, 3, 8, 11), site = factor(c("A", "A", "B", "B")),
    `..row_id_sitecorr` = c(10, 30, 70, 100)
  ), target = "..row_id_sitecorr")
  pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site")
  ))
  expect_equal(pipeop$train(list(task))[[1L]]$truth(), task$truth())
  expect_equal(pipeop$predict(list(task))[[1L]]$truth(), task$truth())
})


test_that("ComBat uses only site and covariate levels observed in the training rows", {
  skip_if_not_installed("neuroCombat")
  set.seed(21)
  n = 80L
  site = factor(rep(c("A", "B", "C", "D"), each = n / 4L))
  data = data.frame(
    x1 = rnorm(n) + as.integer(site),
    x2 = rnorm(n) + 0.5 * as.integer(site),
    xi = sample(1:30, n, replace = TRUE),
    site = site,
    sex = factor(sample(c("F", "M"), n, replace = TRUE), levels = c("F", "M", "X")),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new("combat_levels", data, target = "y")
  train_task = task$clone()$filter(which(site != "D"))
  held_out = task$clone()$filter(which(site == "D"))
  expect_setequal(train_task$levels("site")$site, c("A", "B", "C", "D"))

  pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = c("x1", "x2", "xi")),
    site_correction = list(b1 = list(site = "site", covariates = "sex")),
    method = list(b1 = "combat"),
    combat_unknown = "noop"
  ))
  pipeop$train(list(train_task))
  expect_identical(pipeop$state$per_block$b1$site_lvls, c("A", "B", "C"))

  cols = c("x1", "x2", "xi")
  noop = suppressWarnings(pipeop$predict(list(held_out))[[1L]])
  expect_equal(
    unname(as.matrix(noop$data(cols = cols))),
    unname(as.matrix(held_out$data(cols = cols))) + 0
  )
  expect_identical(noop$feature_types[id == "xi", type], "numeric")

  pipeop$param_set$set_values(combat_unknown = "baseline")
  baseline = suppressWarnings(pipeop$predict(list(held_out))[[1L]])
  expect_gt(max(abs(
    as.matrix(baseline$data(cols = cols)) - as.matrix(held_out$data(cols = cols))
  )), 0.1)

  splits = lapply(levels(site), function(level) which(site == level))
  resampling = mlr3::rsmp("custom")
  resampling$instantiate(task,
    train_sets = lapply(splits, function(rows) setdiff(seq_len(n), rows)),
    test_sets = splits
  )
  graph = mlr3::as_learner(mlr3pipelines::`%>>%`(
    mlr3pipelines::po("sitecorr",
      blocks = list(b1 = c("x1", "x2")),
      site_correction = list(b1 = list(site = "site", covariates = character(0))),
      method = list(b1 = "combat")
    ),
    mlr3::lrn("regr.featureless")
  ))
  result = mlr3::resample(task, graph, resampling)
  expect_equal(nrow(result$errors), 0L)

  missing_reference = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = c("x1", "x2")),
    site_correction = list(b1 = list(site = "site")),
    method = list(b1 = "combat"),
    ref_batch = "D"
  ))
  expect_error(missing_reference$train(list(train_task)), "ref_batch 'D' has no training rows")
  single_site = task$clone()$filter(which(site == "A"))
  expect_error(missing_reference$train(list(single_site)),
    "at least two observed levels in the training rows")
})


test_that("site correction rejects target columns that are read at prediction time", {
  set.seed(22)
  n = 60L
  data = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    site = factor(rep(c("A", "B", "C"), each = n / 3L)),
    y = factor(rep(c("case", "control"), n / 2L))
  )
  task = mlr3::TaskClassif$new("sitecorr_target", data, target = "y")
  make = function(spec, method, ...) {
    PipeOpSiteCorrection$new(param_vals = list(
      blocks = list(b = c("x1", "x2")),
      site_correction = list(b = spec),
      method = list(b = method), ...
    ))
  }
  expect_error(make(c("site", "y"), "partial_corr")$train(list(task)),
    "target column\\(s\\) y cannot be used")
  expect_error(make("y", "partial_corr")$train(list(task)), "target column\\(s\\) y")
  expect_error(make("y", "dir")$train(list(task)), "target column\\(s\\) y")
  expect_error(
    make(list(site = "y", covariates = character(0)), "combat")$train(list(task)),
    "target column\\(s\\) y"
  )

  skip_if_not_installed("neuroCombat")
  covariate = make(list(site = "site", covariates = "y"), "combat")
  covariate$train(list(task))
  reference = suppressWarnings(covariate$predict(list(task))[[1L]])
  flipped_data = data
  flipped_data$y = factor(ifelse(data$y == "case", "control", "case"))
  flipped = mlr3::TaskClassif$new("sitecorr_flipped", flipped_data, target = "y")
  flipped_out = suppressWarnings(covariate$predict(list(flipped))[[1L]])
  expect_equal(flipped_out$data(cols = c("x1", "x2")), reference$data(cols = c("x1", "x2")))
  unlabeled = suppressWarnings(covariate$predict(list(
    mlr3::TaskClassif$new("sitecorr_unlabeled", transform(data, y = factor(NA, levels = levels(data$y))),
      target = "y")
  ))[[1L]])
  expect_equal(unlabeled$data(cols = c("x1", "x2")), reference$data(cols = c("x1", "x2")))
})


test_that("zero_center controls the output location, including for unseen sites", {
  train_task = mlr3::TaskRegr$new("zero_center_train", data.frame(
    x = c(1, 3, 11, 13),
    site = factor(c("A", "A", "B", "B")),
    y = 1:4
  ), target = "y")
  new_task = mlr3::TaskRegr$new("zero_center_new", data.frame(
    x = c(100, 200, 2, 12),
    site = factor(c("new", NA, "A", "B")),
    y = 1:4
  ), target = "y")
  fit = function(zero_center) {
    pipeop = PipeOpSiteCorrection$new(param_vals = list(
      blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
      zero_center = zero_center
    ))
    trained = pipeop$train(list(train_task))[[1L]]
    list(
      train = trained$data(cols = "x")$x,
      replay = pipeop$predict(list(train_task))[[1L]]$data(cols = "x")$x,
      new = pipeop$predict(list(new_task))[[1L]]$data(cols = "x")$x
    )
  }
  grand_mean = 7
  centred = fit(TRUE)
  original = fit(FALSE)

  expect_equal(mean(centred$train), 0)
  expect_equal(mean(original$train), grand_mean)
  expect_equal(centred$replay, centred$train)
  expect_equal(original$replay, original$train)
  expect_equal(original$new, c(100, 200, 7, 7))
  expect_equal(centred$new, c(93, 193, 0, 0))
  expect_equal(centred$new, original$new - grand_mean)
})


test_that("DIR keeps feature names exactly", {
  set.seed(23)
  n = 40L
  data = data.frame(
    `HLA-A` = rnorm(n), `1gene` = rnorm(n), protected = rnorm(n),
    site = factor(rep(c("A", "B"), n / 2L)), y = rnorm(n),
    check.names = FALSE
  )
  task = mlr3::TaskRegr$new("dir_names", data, target = "y")
  features = c("HLA-A", "1gene", "protected")
  for (lambda in c(0, 0.5)) {
    pipeop = PipeOpSiteCorrection$new(param_vals = list(
      blocks = list(b1 = features), site_correction = list(b1 = "site"),
      method = list(b1 = "dir"), lambda = lambda
    ))
    trained = pipeop$train(list(task))[[1L]]
    expect_setequal(trained$feature_names, features)
    expect_named(pipeop$state$per_block$b1$repair_maps, features, ignore.order = TRUE)
    replayed = pipeop$predict(list(task))[[1L]]
    expect_equal(replayed$data(cols = features), trained$data(cols = features))
    if (lambda == 0) {
      expect_equal(trained$data(cols = features), task$data(cols = features))
    }
    new_data = data[1:4, ]
    new_data$`HLA-A` = c(-50, 50, 0, 1)
    new_task = mlr3::TaskRegr$new("dir_names_new", new_data, target = "y")
    predicted = pipeop$predict(list(new_task))[[1L]]$data(cols = features)
    expect_true(all(is.finite(as.matrix(predicted))))
  }
})


test_that("DIR applies the continuous geometric repair without quantization", {
  set.seed(24)
  small = 5L
  big = 55L
  data = data.frame(
    x = c(rnorm(small, mean = 1), rnorm(big)),
    site = factor(rep(c("small", "big"), c(small, big))),
    y = rnorm(small + big)
  )
  task = mlr3::TaskRegr$new("dir_geometric", data, target = "y")
  repair = function(lambda) {
    pipeop = PipeOpSiteCorrection$new(param_vals = list(
      blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
      method = list(b1 = "dir"), lambda = lambda
    ))
    list(pipeop = pipeop, x = pipeop$train(list(task))[[1L]]$data(cols = "x")$x)
  }
  in_big = data$site == "big"

  weak = repair(0.05)$x
  expect_length(unique(weak[in_big]), big)
  expect_identical(order(weak[in_big]), order(data$x[in_big]))
  expect_identical(order(weak[!in_big]), order(data$x[!in_big]))
  expect_lt(max(abs(weak - data$x)), 0.05 * diff(range(data$x)) + 1e-12)
  expect_lt(max(abs(repair(1e-6)$x - data$x)), 1e-4)

  # Explicit formula at the training values of both groups.
  lambda = 0.3
  fitted = repair(lambda)$x
  sorted = split(data$x, data$site)
  cdf = function(values, v) {
    s = sort(values)
    stats::approx(s, (seq_along(s) - 1) / (length(s) - 1), xout = v, ties = mean)$y
  }
  expected = vapply(seq_len(nrow(data)), function(i) {
    p = cdf(sorted[[as.character(data$site[i])]], data$x[i])
    q = vapply(sorted, stats::quantile, numeric(1L), probs = p, type = 7, names = FALSE)
    (1 - lambda) * data$x[i] + lambda * stats::median(q)
  }, numeric(1L))
  expect_equal(fitted, expected, tolerance = 1e-10)

  full = repair(1)$x
  expect_equal(
    unname(stats::quantile(full[in_big], c(0, 0.5, 1))),
    unname(stats::quantile(full[!in_big], c(0, 0.5, 1)))
  )

  tiny = data[c(1:2, 6:60), ]
  tiny_task = mlr3::TaskRegr$new("dir_tiny", tiny, target = "y")
  tiny_pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
    method = list(b1 = "dir"), lambda = 0.5
  ))
  tiny_x = tiny_pipeop$train(list(tiny_task))[[1L]]$data(cols = "x")$x
  expect_length(unique(tiny_x[tiny$site == "big"]), big)
})


test_that("fitted site corrections are frozen at prediction", {
  set.seed(25)
  n = 60L
  data = data.frame(
    x1 = rnorm(n), x2 = rnorm(n),
    site = factor(rep(c("A", "B", "C"), each = n / 3L)), y = rnorm(n)
  )
  data$x1 = data$x1 + as.integer(data$site)
  task = mlr3::TaskRegr$new("sitecorr_frozen", data, target = "y")
  shifted = data[c(5, 25, 45), ]
  shifted$x1 = shifted$x1 + c(10, -10, 3)
  new_task = mlr3::TaskRegr$new("sitecorr_frozen_new", shifted, target = "y")
  methods = c("partial_corr", "dir")
  if (requireNamespace("neuroCombat", quietly = TRUE)) methods = c(methods, "combat")
  for (method in methods) {
    spec = if (identical(method, "combat")) list(site = "site") else "site"
    pipeop = PipeOpSiteCorrection$new(param_vals = list(
      blocks = list(b1 = c("x1", "x2")), site_correction = list(b1 = spec),
      method = list(b1 = method)
    ))
    pipeop$train(list(task))
    state = pipeop$state
    batch = pipeop$predict(list(new_task))[[1L]]$data(cols = c("x1", "x2"))
    expect_identical(pipeop$state, state)
    single = lapply(new_task$row_ids, function(id) {
      pipeop$predict(list(new_task$clone()$filter(id)))[[1L]]$data(cols = c("x1", "x2"))
    })
    expect_equal(data.table::rbindlist(single), batch, info = method)
  }

  # Site columns without a role must still be present at prediction.
  roleless = task$clone()
  roleless$col_roles$feature = c("x1", "x2")
  pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = c("x1", "x2")), site_correction = list(b1 = "site"),
    method = list(b1 = "dir")
  ))
  pipeop$train(list(roleless))
  no_site = mlr3::TaskRegr$new("sitecorr_no_site", shifted[c("x1", "x2", "y")], target = "y")
  expect_error(pipeop$predict(list(no_site)), "lacks site or protected column\\(s\\): site")
})


test_that("partial_corr warns when a site has a single fitting row", {
  set.seed(26)
  n = 41L
  data = data.frame(
    x1 = rnorm(n), x2 = rnorm(n),
    site = factor(c(rep("A", 20L), rep("B", 20L), "C")),
    age = rnorm(n, 50, 8), y = rnorm(n)
  )
  task = mlr3::TaskRegr$new("sitecorr_singleton", data, target = "y")
  make = function(spec, ...) {
    PipeOpSiteCorrection$new(param_vals = list(
      blocks = list(b1 = c("x1", "x2")), site_correction = list(b1 = spec), ...
    ))
  }
  single = make("site")
  expect_warning(
    {
      trained = single$train(list(task))[[1L]]
    },
    "reproduced exactly.*site = C"
  )
  expect_equal(unlist(trained$data(rows = n, cols = c("x1", "x2"))),
    colMeans(data[c("x1", "x2")]))
  expect_warning(make(c("site", "age"))$train(list(task)), "reproduced exactly.*site = C")

  expect_no_warning({
    ridge = make("site", regularization = 1)$train(list(task))[[1L]]
  })
  expect_false(isTRUE(all.equal(
    unlist(ridge$data(rows = n, cols = c("x1", "x2"))),
    colMeans(data[c("x1", "x2")])
  )))
  balanced = task$clone()$filter(seq_len(40L))
  expect_no_warning(make("site")$train(list(balanced)))
})


test_that("subgroup uses row ids or columns and is resampling-safe", {
  set.seed(27)
  n = 100L
  dx = factor(rep(c("control", "case"), each = n / 2L))
  site = factor(sample(c("A", "B"), n, replace = TRUE))
  data = data.frame(
    x = rnorm(n) + 2 * (site == "B") + 3 * (dx == "case") * (site == "B"),
    site = site, dx = dx, is_control = dx == "control", y = rnorm(n)
  )
  task = mlr3::TaskRegr$new("sitecorr_subgroup", data, target = "y")
  task$set_col_roles(c("dx", "is_control"), character(0))
  controls = which(dx == "control")
  controls_only = function(rows) {
    unname(stats::coef(stats::lm(x ~ site, data = data[intersect(rows, controls), ]))[2L])
  }
  site_effect = function(subgroup, rows) {
    pipeop = PipeOpSiteCorrection$new(param_vals = list(
      blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
      subgroup = subgroup
    ))
    out = pipeop$train(list(task$clone()$filter(rows)))[[1L]]
    list(beta = unname(pipeop$state$per_block$b1$beta[2L, 1L]), task = out)
  }
  rows = sort(sample(n, 80L))
  expect_equal(site_effect(controls, rows)$beta, controls_only(rows))
  expect_equal(site_effect("is_control", rows)$beta, controls_only(rows))
  by_level = site_effect(list(column = "dx", values = "control"), rows)
  expect_equal(by_level$beta, controls_only(rows))
  expect_identical(by_level$task$feature_names, "x")

  graph = mlr3::as_learner(mlr3pipelines::`%>>%`(
    mlr3pipelines::po("sitecorr",
      blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
      subgroup = controls
    ),
    mlr3::lrn("regr.featureless")
  ))
  result = mlr3::resample(task, graph, mlr3::rsmp("cv", folds = 3L), store_models = TRUE)
  expect_equal(nrow(result$errors), 0L)
  for (i in seq_len(3L)) {
    beta = result$learners[[i]]$model$sitecorr$per_block$b1$beta[2L, 1L]
    expect_equal(unname(beta), controls_only(result$resampling$train_set(i)))
  }

  unused = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
    method = list(b1 = "dir"), subgroup = controls
  ))
  expect_error(unused$train(list(task)), "only used by 'partial_corr'")
  not_logical = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"), subgroup = "dx"
  ))
  expect_error(not_logical$train(list(task)), "must be logical")

  # Positions of a task whose row ids are not 1..n are not row ids.
  shifted = mlr3::TaskRegr$new("sitecorr_subgroup_ids",
    mlr3::as_data_backend(cbind(data, id = 1000L + seq_len(n)), primary_key = "id"),
    target = "y"
  )
  shifted$set_col_roles(c("dx", "is_control"), character(0))
  positions = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
    subgroup = c(controls, 1000L + controls[1:5])
  ))
  expect_error(positions$train(list(shifted)), "do not exist in the task's backend")
  by_id = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = "x"), site_correction = list(b1 = "site"),
    subgroup = 1000L + controls
  ))
  by_id$train(list(shifted$clone()$filter(1000L + rows)))
  expect_equal(unname(by_id$state$per_block$b1$beta[2L, 1L]), controls_only(rows))
})


test_that("site correction output tasks keep column information consistent", {
  set.seed(28)
  n = 60L
  data = data.frame(
    x = rnorm(n), xi = sample(1:20, n, replace = TRUE),
    site = factor(rep(c("A", "B", "C"), each = n / 3L)),
    g = rep(seq_len(12L), 5L), y = factor(rep(c("a", "b", "b"), n / 3L))
  )
  task = mlr3::TaskClassif$new("sitecorr_colinfo", data, target = "y")
  task$set_col_roles("g", "group")
  pipeop = PipeOpSiteCorrection$new(param_vals = list(
    blocks = list(b1 = c("x", "xi")), site_correction = list(b1 = "site")
  ))
  for (out in list(pipeop$train(list(task))[[1L]], pipeop$predict(list(task))[[1L]])) {
    expect_true(setequal(out$col_info$id, out$backend$colnames))
    expect_identical(out$feature_names, c("x", "xi"))
    expect_identical(out$feature_types$type, c("numeric", "numeric"))
    expect_identical(out$col_roles$group, "g")
    expect_equal(out$truth(), task$truth())
    expect_true(is.double(out$data(cols = "xi")$xi))
    expect_silent(out$data(cols = out$backend$primary_key))
  }

  graph = mlr3pipelines::`%>>%`(
    mlr3pipelines::po("sitecorr", blocks = list(b1 = c("x", "xi")),
      site_correction = list(b1 = "site")),
    mlr3pipelines::po("scale", affect_columns = mlr3pipelines::selector_type("numeric"))
  )
  scaled = graph$train(task)[[1L]]
  expect_equal(unname(colMeans(scaled$data(cols = c("x", "xi")))), c(0, 0))

  balancing = mlr3pipelines::`%>>%`(
    mlr3pipelines::po("sitecorr", blocks = list(b1 = c("x", "xi")),
      site_correction = list(b1 = "site")),
    mlr3pipelines::po("classbalancing", adjust = "minor", reference = "major")
  )
  # Row-binding (oversampling) relies on consistent keys and column types.
  ungrouped = task$clone()
  ungrouped$col_roles$group = character(0)
  balanced = balancing$train(ungrouped)[[1L]]
  expect_equal(as.integer(table(balanced$truth())), c(40L, 40L))
})
