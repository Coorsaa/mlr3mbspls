test_that(".mbspls_state_from_env prefers the requested run-specific state", {
  env = new.env(parent = emptyenv())

  st_old = list(
    run_id = "run_old",
    blocks = list(b1 = "x1"),
    weights = list(LC_01 = list(b1 = stats::setNames(1, "x1")))
  )
  st_new = list(
    run_id = "run_new",
    blocks = list(b1 = "x1"),
    weights = list(LC_01 = list(b1 = stats::setNames(2, "x1")))
  )

  env$mbspls_states = list(run_old = st_old, run_new = st_new)
  env$mbspls_state = st_new

  got = mlr3mbspls:::.mbspls_state_from_env(env, run_id = "run_old", where = "log_env")
  expect_identical(got$run_id, "run_old")
  expect_equal(unname(got$weights[[1L]]$b1), 1)
})

test_that("log_env_store_last does not infer an unrelated run_id", {
  env = new.env(parent = emptyenv())
  env$mbspls_state = list(run_id = "other_run")

  mlr3mbspls:::log_env_store_last(env, list(flag = TRUE), run_id = NULL)

  expect_true(is.list(env$last))
  expect_true(isTRUE(env$last$flag))
  expect_true(is.null(env$mbspls_last))
})

test_that("log_env_store_state warns by default and respects env overrides", {
  old_payload = list(run_id = "run_old")
  new_payload = list(run_id = "run_new")

  env_default = new.env(parent = emptyenv())
  mlr3mbspls:::log_env_store_state(env_default, old_payload, warn_overwrite = TRUE)
  expect_warning(
    mlr3mbspls:::log_env_store_state(env_default, new_payload, warn_overwrite = TRUE),
    "will be overwritten"
  )

  env_silent = new.env(parent = emptyenv())
  env_silent$warn_overwrite = FALSE
  mlr3mbspls:::log_env_store_state(env_silent, old_payload, warn_overwrite = TRUE)
  expect_no_warning(
    mlr3mbspls:::log_env_store_state(env_silent, new_payload, warn_overwrite = TRUE)
  )
})
