test_that("target_label_filter keeps >= 2 levels with invert = FALSE", {
  set.seed(1)

  df = data.frame(
    x = rnorm(60),
    y = factor(c(rep("A", 30), rep("B", 30)))
  )

  tsk = mlr3::TaskClassif$new("toy", backend = df, target = "y")
  po = PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y",
    labels = c("A", "B"), # keep both labels
    invert = FALSE
  ))

  # Simulate a single-observed-class split by filtering rows before PipeOp
  tsk_A = tsk$clone(deep = TRUE)
  tsk_A$filter(which(tsk_A$truth() == "A"))

  out = po$train(list(tsk_A))[[1L]]
  expect_true(is.factor(out$truth()))
  expect_setequal(levels(out$truth()), c("A", "B")) # >= 2 levels (labels)
  expect_true(all(out$truth() == "A")) # but observed is single class
})

test_that("target_label_filter keeps >= 2 levels with invert = TRUE", {
  set.seed(2)

  df = data.frame(
    x = rnorm(90),
    y = factor(rep(c("A", "B", "C"), each = 30))
  )
  tsk = mlr3::TaskClassif$new("toy2", backend = df, target = "y")

  # Drop label "A" -> others = {B, C}; now simulate that only B remains
  po = PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y",
    labels = "A",
    invert = TRUE
  ))

  tsk_B = tsk$clone(deep = TRUE)
  tsk_B$filter(which(tsk_B$truth() == "B"))

  out = po$train(list(tsk_B))[[1L]]
  expect_true(is.factor(out$truth()))
  # Others set is {B, C}; padded to ensure >=2 levels
  expect_setequal(levels(out$truth()), c("B", "C"))
  expect_true(all(out$truth() == "B"))
})

test_that("drop_stratum removes only the role, not features/targets", {
  df = data.frame(
    x1 = rnorm(10),
    x2 = rnorm(10),
    s  = sample(letters[1:2], 10, replace = TRUE),
    y  = factor(sample(c("A", "B"), 10, replace = TRUE))
  )
  tsk = mlr3::TaskClassif$new("toy3", backend = df, target = "y")
  tsk$set_col_roles("s", roles = "stratum")

  po = PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y", labels = c("A", "B"), drop_stratum = TRUE
  ))
  out = po$train(list(tsk))[[1L]]

  expect_false("s" %in% out$col_roles$stratum)
  expect_true(all(c("x1", "x2") %in% out$feature_names))
  expect_true("y" %in% out$target_names)
})

test_that("drop_unused_levels works", {
  df = data.frame(
    x = rnorm(20),
    f = factor(rep(c("u", "v", "w", "w"), 5L), levels = c("u", "v", "w", "unused")),
    y = factor(rep(c("A", "B", "C", "C"), 5L))
  )
  tsk = mlr3::TaskClassif$new("toy4", backend = df, target = "y")

  po = PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y", labels = c("A", "B"), drop_unused_levels = TRUE
  ))

  # Simulate single observed class "A"
  tsk_A = tsk$clone(deep = TRUE)
  tsk_A$filter(which(tsk_A$truth() == "A"))

  out = po$train(list(tsk_A))[[1L]]
  expect_true(is.factor(out$truth()))
  expect_setequal(levels(out$truth()), c("A", "B")) # C dropped
  expect_true(all(out$truth() == "A"))
  expect_identical(out$levels("f")$f, "u")

  kept = PipeOpTargetLabelFilter$new(param_vals = list(
    target = "y", labels = c("A", "B"), drop_unused_levels = FALSE
  ))
  out = kept$train(list(tsk_A))[[1L]]
  expect_identical(out$levels("f")$f, c("u", "v", "w", "unused"))
  expect_identical(kept$predict(list(tsk))[[1L]]$levels("f")$f, c("u", "v", "w", "unused"))
})


test_that("target filtering never selects prediction rows using held-out labels", {
  task = mlr3::TaskClassif$new("label_filter_train", data.frame(
    x = 1:6, y = factor(rep(c("A", "B", "C"), 2))
  ), target = "y")
  pipeop = PipeOpTargetLabelFilter$new(param_vals = list(labels = c("A", "B")))
  trained = pipeop$train(list(task))[[1L]]
  expect_equal(trained$nrow, 4L)
  predicted = pipeop$predict(list(task))[[1L]]
  expect_equal(predicted$row_ids, task$row_ids)
  expect_equal(predicted$truth(), task$truth())

  unlabeled = mlr3::TaskClassif$new("label_filter_unlabeled", data.frame(
    x = 1:3, y = factor(rep(NA_character_, 3), levels = c("A", "B", "C"))
  ), target = "y")
  expect_equal(pipeop$predict(list(unlabeled))[[1L]]$row_ids, unlabeled$row_ids)
})


test_that("drop_stratum preserves unrelated grouping roles", {
  task = mlr3::TaskClassif$new("label_filter_roles", data.frame(
    x = 1:8, site = factor(rep(c("a", "b"), each = 4)),
    y = factor(rep(c("A", "B"), 4))
  ), target = "y")
  task$set_col_roles("site", roles = c("stratum", "group"))
  pipeop = PipeOpTargetLabelFilter$new(param_vals = list(
    labels = c("A", "B"), drop_stratum = TRUE
  ))
  trained = pipeop$train(list(task))[[1L]]
  expect_false("site" %in% trained$col_roles$stratum)
  expect_true("site" %in% trained$col_roles$group)
})


test_that("single-label filtering keeps the padded target levels next to factor features", {
  set.seed(3)
  n = 60L
  data = data.frame(
    x = rnorm(n),
    sex = factor(c(rep(c("F", "M"), 15L), rep(c("F", "M", "X"), 10L)), levels = c("F", "M", "X")),
    site = factor(sample(c("a", "b"), n, replace = TRUE), levels = c("a", "b", "c")),
    dx = factor(rep(c("control", "case"), each = n / 2L), levels = c("control", "case"))
  )
  task = mlr3::TaskClassif$new("label_filter_factors", data, target = "dx")

  controls = PipeOpTargetLabelFilter$new(param_vals = list(labels = "control"))
  out = controls$train(list(task))[[1L]]
  expect_identical(out$class_names, c("control", "case"))
  expect_true(all(out$truth() == "control"))
  expect_identical(out$levels("sex")$sex, c("F", "M"))
  expect_identical(out$levels("site")$site, c("a", "b"))

  inverted = PipeOpTargetLabelFilter$new(param_vals = list(labels = "case", invert = TRUE))
  expect_identical(inverted$train(list(task))[[1L]]$class_names, c("control", "case"))

  unobserved = PipeOpTargetLabelFilter$new(param_vals = list(labels = c("control", "zzz")))
  expect_identical(unobserved$train(list(task))[[1L]]$class_names, c("control", "zzz"))

  three = mlr3::TaskClassif$new("label_filter_three", data.frame(
    x = rnorm(90L),
    f = factor(c(rep(c("u", "v"), 30L), rep("w", 30L))),
    y = factor(rep(c("A", "B", "C"), each = 30L))
  ), target = "y")
  unobserved_c = PipeOpTargetLabelFilter$new(param_vals = list(labels = c("A", "B", "C")))
  observed_ab = three$clone()$filter(1:60)
  expect_identical(unobserved_c$train(list(observed_ab))[[1L]]$class_names, c("A", "B", "C"))
})


test_that("training factor levels are re-applied at prediction", {
  set.seed(4)
  data = data.frame(
    x = rnorm(60L),
    sex = factor(c(rep(c("F", "M"), 15L), rep(c("F", "M", "X"), 10L))),
    dx = factor(rep(c("control", "case"), each = 30L), levels = c("control", "case"))
  )
  task = mlr3::TaskClassif$new("label_filter_frozen", data, target = "dx")
  pipeop = PipeOpTargetLabelFilter$new(param_vals = list(labels = "control"))
  pipeop$train(list(task))
  expect_identical(pipeop$state$factor_levels, list(sex = c("F", "M")))

  predicted = pipeop$predict(list(task))[[1L]]
  expect_identical(predicted$row_ids, task$row_ids)
  expect_identical(predicted$levels("sex")$sex, c("F", "M"))
  expect_identical(predicted$levels("dx")$dx, c("control", "case"))
  sex = predicted$data(cols = "sex")$sex
  expect_identical(is.na(sex), data$sex == "X")

  single = PipeOpTargetLabelFilter$new(param_vals = list(labels = "control"))
  learner = mlr3::as_learner(mlr3pipelines::`%>>%`(single, mlr3::lrn("classif.featureless")))
  learner$train(task)
  expect_length(learner$predict(task)$response, 60L)

  three = mlr3::TaskClassif$new("label_filter_graph", data.frame(
    x = rnorm(90L),
    f = factor(c(rep(c("u", "v"), 30L), rep("w", 30L))),
    y = factor(rep(c("A", "B", "C"), each = 30L))
  ), target = "y")
  graph = mlr3::as_learner(mlr3pipelines::`%>>%`(
    PipeOpTargetLabelFilter$new(param_vals = list(labels = c("A", "B"))),
    mlr3::lrn("classif.rpart")
  ))
  graph$train(three)
  expect_length(graph$predict(three)$response, 90L)
})


test_that("filtering to fewer observed classes updates the class property", {
  set.seed(5)
  data = data.frame(
    x = rnorm(90L),
    f = factor(c(rep(c("u", "v"), 30L), rep("w", 30L)), levels = c("u", "v", "w", "z")),
    y = factor(rep(c("A", "B", "C"), each = 30L))
  )
  task = mlr3::TaskClassif$new("label_filter_property", data, target = "y")
  expect_true("multiclass" %in% task$properties)

  kept = PipeOpTargetLabelFilter$new(param_vals = list(
    labels = c("A", "B"), drop_unused_levels = FALSE
  ))
  out = kept$train(list(task$clone()))[[1L]]
  expect_true("twoclass" %in% out$properties)
  expect_false("multiclass" %in% out$properties)
  expect_identical(out$class_names, c("A", "B"))
  expect_identical(out$levels("f")$f, c("u", "v", "w", "z"))

  dropped = PipeOpTargetLabelFilter$new(param_vals = list(labels = c("A", "B")))
  out = dropped$train(list(task$clone()))[[1L]]
  expect_true("twoclass" %in% out$properties)
  expect_false("multiclass" %in% out$properties)
  expect_identical(out$levels("f")$f, c("u", "v"))

  inverted = PipeOpTargetLabelFilter$new(param_vals = list(labels = "C", invert = TRUE))
  expect_true("twoclass" %in% inverted$train(list(task$clone()))[[1L]]$properties)

  skip_if_not_installed("mlr3learners")
  learner = mlr3::as_learner(mlr3pipelines::`%>>%`(
    PipeOpTargetLabelFilter$new(param_vals = list(labels = c("A", "B"))),
    mlr3::lrn("classif.log_reg")
  ))
  learner$train(task)
  binary = data[1:60, ]
  binary$y = factor(as.character(binary$y))
  prediction = learner$predict(mlr3::TaskClassif$new("label_filter_binary", binary, target = "y"))
  expect_length(prediction$response, 60L)
  expect_true(all(prediction$response %in% c("A", "B")))
})


test_that("padded target level sets update the class property", {
  skip_if(utils::packageVersion("mlr3") < "1.7.0",
    "mlr3 < 1.7.0 cannot update the property of a padded level set")
  task = mlr3::TaskClassif$new("label_filter_padded", data.frame(
    x = rnorm(90L),
    f = factor(c(rep(c("u", "v"), 30L), rep("w", 30L))),
    y = factor(rep(c("A", "B", "C"), each = 30L))
  ), target = "y")
  out = PipeOpTargetLabelFilter$new(param_vals = list(labels = "A"))$train(list(task))[[1L]]
  expect_identical(out$class_names, c("A", "B"))
  expect_true("twoclass" %in% out$properties)
  expect_false("multiclass" %in% out$properties)
})


test_that("factor columns without a feature role keep their levels and values", {
  set.seed(6)
  data = data.frame(
    x = rnorm(60L),
    f = factor(rep(c("u", "v", "w", "v"), 15L)),
    site = factor(rep(c("s1", "s2", "s3"), each = 20L)),
    y = factor(rep(c("A", "B"), 30L))
  )
  task = mlr3::TaskClassif$new("label_filter_site", data, target = "y")
  task$set_col_roles("site", roles = "group")

  pipeop = PipeOpTargetLabelFilter$new(param_vals = list(labels = "A"))
  out = pipeop$train(list(task$clone()$filter(1:40)))[[1L]]
  expect_identical(out$levels("site")$site, c("s1", "s2", "s3"))
  expect_identical(out$levels("f")$f, c("u", "w"))
  expect_identical(names(pipeop$state$factor_levels), "f")

  predicted = pipeop$predict(list(task))[[1L]]
  expect_identical(predicted$data(cols = "site")$site, data$site)
  expect_identical(is.na(predicted$data(cols = "f")$f), data$f == "v")
})
