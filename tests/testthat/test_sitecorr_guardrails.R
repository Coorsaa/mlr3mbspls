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
