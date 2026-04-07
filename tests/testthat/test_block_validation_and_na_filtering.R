test_that("TaskMultiBlock rejects overlapping block definitions", {
  dt = data.table::data.table(
    x1 = rnorm(12),
    x2 = rnorm(12),
    y = rnorm(12)
  )

  expect_error(
    TaskMultiBlock(
      dt,
      blocks = list(a = c("x1", "x2"), b = c("x2")),
      target = "y",
      task_type = "regr"
    ),
    "disjoint across blocks"
  )
})


test_that("mb_resolve_blocks drops all-NA and zero-variance numeric columns", {
  dt = data.table::data.table(
    x_ok = rnorm(10),
    x_const = rep(1, 10),
    x_all_na = rep(NA_real_, 10),
    x_some_na = c(rnorm(9), NA_real_)
  )

  got = mlr3mbspls:::mb_resolve_blocks(
    dt,
    blocks = list(a = c("x_ok", "x_const", "x_all_na", "x_some_na")),
    numeric_only = TRUE,
    non_constant = TRUE
  )

  expect_named(got, "a")
  expect_setequal(got$a, c("x_ok", "x_some_na"))
})



test_that("direct PipeOps reject overlapping block definitions", {
  expect_error(
    PipeOpMBsPLS$new(blocks = list(a = c("x1", "x2"), b = c("x2"))),
    "disjoint across blocks"
  )
  expect_error(
    PipeOpMBsPCA$new(blocks = list(a = c("x1", "x2"), b = c("x2"))),
    "disjoint across blocks"
  )
  expect_error(
    PipeOpMBsPLSXY$new(blocks = list(a = c("x1", "x2"), b = c("x2"))),
    "disjoint across blocks"
  )
})
