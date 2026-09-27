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


make_resolver_task = function(n = 40L, seed = 71L) {
  set.seed(seed)
  data.frame(
    age = rnorm(n),
    a1 = rnorm(n), a2 = rnorm(n),
    age.onset = rnorm(n),
    b1 = rnorm(n), b2 = rnorm(n),
    sex = factor(sample(c("f", "m", "x"), n, replace = TRUE)),
    sex.hormone = rnorm(n),
    y = rnorm(n)
  )
}

resolver_pipeops = function(blocks) {
  list(
    mbspls = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(c_A = 1.2, c_B = 1.2)),
    mbsplsxy = PipeOpMBsPLSXY$new(blocks = blocks),
    mbspca = PipeOpMBsPCA$new(blocks = blocks)
  )
}

resolved_blocks = function(pipeop) {
  pipeop$state$blocks %||% pipeop$state$blocks_x
}


test_that("multi-block PipeOps never let a dropped declared name claim another block's column", {
  df = make_resolver_task()
  task = mlr3::TaskRegr$new("resolver_drop", df[, setdiff(names(df), c("sex", "sex.hormone"))], target = "y")
  blocks = list(A = c("age", "a1", "a2"), B = c("age.onset", "b1", "b2"))

  for (pipeop in resolver_pipeops(blocks)) {
    graph = po("select", selector = selector_invert(selector_name("age"))) %>>% pipeop
    graph$train(task)
    fitted = resolved_blocks(graph$pipeops[[pipeop$id]])
    expect_identical(fitted$A, c("a1", "a2"), info = pipeop$id)
    expect_identical(fitted$B, c("age.onset", "b1", "b2"), info = pipeop$id)
  }
})


test_that("multi-block PipeOps resolve encoded factors next to declared prefixed features", {
  df = make_resolver_task(seed = 72L)
  task = mlr3::TaskRegr$new("resolver_encode", df[, setdiff(names(df), c("age", "age.onset"))], target = "y")
  blocks = list(A = c("sex", "a1", "a2"), B = c("sex.hormone", "b1", "b2"))

  for (pipeop in resolver_pipeops(blocks)) {
    graph = po("encode", method = "treatment") %>>% pipeop
    graph$train(task)
    fitted = resolved_blocks(graph$pipeops[[pipeop$id]])
    expect_setequal(fitted$A, c("sex.m", "sex.x", "a1", "a2"))
    expect_identical(fitted$B, c("sex.hormone", "b1", "b2"), info = pipeop$id)
  }
})
