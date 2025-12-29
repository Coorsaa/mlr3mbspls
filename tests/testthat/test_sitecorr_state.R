test_that("PipeOpSiteCorrection - partial_corr fits, stores state, and predicts", {
  set.seed(1)

  n = 60
  x1 = rnorm(n)
  x2 = rnorm(n)
  x3 = rnorm(n)
  x4 = rnorm(n)
  x5 = rnorm(n)
  x6 = rnorm(n)

  site = factor(sample(c("A", "B", "C"), n, replace = TRUE))
  age = rnorm(n, mean = 50, sd = 10)
  sex = factor(sample(c("F", "M"), n, replace = TRUE))
  bmi = rnorm(n, mean = 25, sd = 4)

  y = 0.2 * x1 - 0.1 * x4 + rnorm(n, sd = 0.5)

  df = data.frame(x1, x2, x3, x4, x5, x6, site, age, sex, bmi, y)
  task = mlr3::TaskRegr$new(id = "mb_site", backend = df, target = "y")

  blocks = list(
    b1 = c("x1", "x2", "x3"),
    b2 = c("x4", "x5", "x6")
  )
  site_spec = list(
    # For `partial_corr`, the spec is a *character vector* of columns
    # (categorical site and/or numeric covariates).
    b1 = c("site", "age", "sex", "bmi"),
    b2 = c("site", "age", "sex", "bmi")
  )
  method_spec = list(b1 = "partial_corr", b2 = "partial_corr")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = blocks,
      site_correction = site_spec,
      method = method_spec,
      keep_site_col = TRUE
    )
  )

  out_train = po$train(list(task))[[1L]]
  st = po$state

  expect_true(is.list(st))
  expect_true(is.list(st$per_block))
  expect_setequal(names(st$per_block), names(blocks))
  expect_equal(st$per_block$b1$method, "partial_corr")
  expect_equal(st$per_block$b2$method, "partial_corr")

  # keep_site_col = TRUE keeps site and covariates as features
  expect_true(all(c("site", "age", "sex", "bmi") %in% out_train$feature_names))

  out_pred = po$predict(list(task))[[1L]]
  expect_s3_class(out_pred, "Task")
  expect_equal(out_pred$nrow, task$nrow)
})

test_that("PipeOpSiteCorrection - keep_site_col=FALSE drops site and covariates", {
  set.seed(1)

  n = 40
  df = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n),
    site = factor(sample(c("A", "B"), n, replace = TRUE)),
    age = rnorm(n, mean = 50, sd = 10),
    sex = factor(sample(c("F", "M"), n, replace = TRUE)),
    bmi = rnorm(n, mean = 25, sd = 4),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "mb_site2", backend = df, target = "y")

  blocks = list(b1 = c("x1", "x2", "x3"))
  site_spec = list(b1 = c("site", "age", "sex", "bmi"))
  method_spec = list(b1 = "partial_corr")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = blocks,
      site_correction = site_spec,
      method = method_spec,
      keep_site_col = FALSE
    )
  )

  out_train = po$train(list(task))[[1L]]
  expect_false(any(c("site", "age", "sex", "bmi") %in% out_train$feature_names))
})

test_that("PipeOpSiteCorrection - combat method (if installed)", {
  testthat::skip_if_not_installed("neuroCombat")

  set.seed(1)
  n = 50
  df = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n),
    site = factor(sample(c("A", "B"), n, replace = TRUE)),
    age = rnorm(n, mean = 50, sd = 10),
    sex = factor(sample(c("F", "M"), n, replace = TRUE)),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "mb_combat", backend = df, target = "y")

  blocks = list(b1 = c("x1", "x2", "x3"))
  site_spec = list(b1 = list(site = "site", covariates = c("age", "sex")))
  method_spec = list(b1 = "combat")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = blocks,
      site_correction = site_spec,
      method = method_spec,
      keep_site_col = TRUE
    )
  )

  out_train = po$train(list(task))[[1L]]
  expect_s3_class(out_train, "Task")
  expect_equal(po$state$per_block$b1$method, "combat")
})

test_that("PipeOpSiteCorrection - dir method (if installed)", {
  testthat::skip_if_not_installed("fairmodels")

  set.seed(1)
  n = 50
  df = data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n),
    prot = factor(sample(c("p0", "p1"), n, replace = TRUE)),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "mb_dir", backend = df, target = "y")

  blocks = list(b1 = c("x1", "x2", "x3"))
  site_spec = list(b1 = "prot")
  method_spec = list(b1 = "dir")

  po = PipeOpSiteCorrection$new(
    param_vals = list(
      blocks = blocks,
      site_correction = site_spec,
      method = method_spec,
      keep_site_col = TRUE
    )
  )

  out_train = po$train(list(task))[[1L]]
  expect_s3_class(out_train, "Task")
  expect_equal(po$state$per_block$b1$method, "dir")
})
