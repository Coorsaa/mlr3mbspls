test_that("PipeOpSiteCorrection - dir method errors when protected attribute has only one level", {
  testthat::skip_if_not_installed("fairmodels")

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
  testthat::skip_if_not_installed("fairmodels")

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
  skip_if_not_installed("fairmodels")
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
