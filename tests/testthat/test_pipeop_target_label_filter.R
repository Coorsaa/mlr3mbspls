test_that("target_label_filter keeps >= 2 levels with invert = FALSE", {
  library(mlr3)
  set.seed(1)

  df <- data.frame(
    x = rnorm(60),
    y = factor(c(rep("A", 30), rep("B", 30)))
  )

  tsk <- TaskClassif$new("toy", backend = df, target = "y")
  po  <- PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y",
    labels = c("A", "B"),  # keep both labels
    invert = FALSE
  ))

  # Simulate a single-observed-class split by filtering rows before PipeOp
  tsk_A <- tsk$clone(deep = TRUE)
  tsk_A$filter(which(tsk_A$truth() == "A"))

  out <- po$train(list(tsk_A))[[1L]]
  expect_true(is.factor(out$truth()))
  expect_setequal(levels(out$truth()), c("A", "B"))     # >= 2 levels (labels)
  expect_true(all(out$truth() == "A"))                   # but observed is single class
})

test_that("target_label_filter keeps >= 2 levels with invert = TRUE", {
  library(mlr3)
  set.seed(2)

  df <- data.frame(
    x = rnorm(90),
    y = factor(rep(c("A", "B", "C"), each = 30))
  )
  tsk <- TaskClassif$new("toy2", backend = df, target = "y")

  # Drop label "A" -> others = {B, C}; now simulate that only B remains
  po <- PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y",
    labels = "A",
    invert = TRUE
  ))

  tsk_B <- tsk$clone(deep = TRUE)
  tsk_B$filter(which(tsk_B$truth() == "B"))

  out <- po$train(list(tsk_B))[[1L]]
  expect_true(is.factor(out$truth()))
  # Others set is {B, C}; padded to ensure >=2 levels
  expect_setequal(levels(out$truth()), c("B", "C"))
  expect_true(all(out$truth() == "B"))
})

test_that("drop_stratum removes only the role, not features/targets", {
  library(mlr3)

  df <- data.frame(
    x1 = rnorm(10),
    x2 = rnorm(10),
    s  = sample(letters[1:2], 10, replace = TRUE),
    y  = factor(sample(c("A", "B"), 10, replace = TRUE))
  )
  tsk <- TaskClassif$new("toy3", backend = df, target = "y")
  tsk$set_col_roles("s", roles = "stratum")

  po <- PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y", labels = c("A", "B"), drop_stratum = TRUE
  ))
  out <- po$train(list(tsk))[[1L]]

  expect_false("s" %in% out$col_roles$stratum)
  expect_true(all(c("x1", "x2") %in% out$feature_names))
  expect_true("y" %in% out$target_names)
})

test_that("drop_unused_levels works", {
  library(mlr3)

  df <- data.frame(
    x = rnorm(20),
    y = factor(rep(c("A", "B", "C"), length.out = 20))
  )
  tsk <- TaskClassif$new("toy4", backend = df, target = "y")

  po <- PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y", labels = c("A", "B"), drop_unused_levels = TRUE
  ))

  # Simulate single observed class "A"
  tsk_A <- tsk$clone(deep = TRUE)
  tsk_A$filter(which(tsk_A$truth() == "A"))

  out <- po$train(list(tsk_A))[[1L]]
  expect_true(is.factor(out$truth()))
  expect_setequal(levels(out$truth()), c("A", "B"))  # C dropped
  expect_true(all(out$truth() == "A"))
})
