test_that("PipeOpBlockScaling preserves supervised targets on train and predict", {
  task = task_multiblock_synthetic(task_type = "classif", n = 30L, seed = 55L)
  blocks = task$block_features()

  po = PipeOpBlockScaling$new(
    param_vals = list(
      blocks = blocks,
      method = "unit_ssq"
    )
  )

  out_train = po$train(list(task))[[1L]]
  expect_true(inherits(out_train, "TaskClassif"))
  expect_equal(out_train$target_names, task$target_names)
  expect_true(all(task$target_names %in% out_train$backend$colnames))
  expect_equal(out_train$truth(), task$truth())
  expect_equal(out_train$class_names, task$class_names)

  out_pred = po$predict(list(task))[[1L]]
  expect_true(inherits(out_pred, "TaskClassif"))
  expect_equal(out_pred$target_names, task$target_names)
  expect_true(all(task$target_names %in% out_pred$backend$colnames))
  expect_equal(out_pred$truth(), task$truth())
  expect_equal(out_pred$class_names, task$class_names)
})


test_that("PipeOpBlockScaling preserves features that collide with the internal row-id name", {
  n = 24L
  df = data.frame(
    `..row_id_blockscale` = rnorm(n),
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "blockscale_collision", backend = df, target = "y")

  po = PipeOpBlockScaling$new(
    param_vals = list(
      blocks = list(b1 = c("..row_id_blockscale", "x1", "x2")),
      method = "unit_ssq"
    )
  )

  out = po$train(list(task))[[1L]]
  expect_true("..row_id_blockscale" %in% out$feature_names)
})

test_that("PipeOpBlockScaling freezes feature scaling and fails unsafe inputs", {
  train_data = data.frame(
    x1 = c(1, 2, 4, 8),
    x2 = c(10, 12, 18, 30),
    y = 1:4
  )
  predict_data = data.frame(
    x1 = c(20, 30),
    x2 = c(40, 60),
    y = 1:2
  )
  train_task = mlr3::TaskRegr$new("scale_train", train_data, target = "y")
  predict_task = mlr3::TaskRegr$new("scale_predict", predict_data,
    target = "y")
  pipeop = PipeOpBlockScaling$new(param_vals = list(
    blocks = list(block = c("x1", "x2")),
    method = "feature_zscore",
    divide_by_sqrt_p = FALSE
  ))

  pipeop$train(list(train_task))
  predicted = pipeop$predict(list(predict_task))[[1L]]
  expected = sweep(as.matrix(predict_data[c("x1", "x2")]), 2,
    colMeans(train_data[c("x1", "x2")]), "-")
  expected = sweep(expected, 2,
    vapply(train_data[c("x1", "x2")], stats::sd, numeric(1L)), "/")
  expect_equal(
    unname(as.matrix(predicted$data(cols = c("x1", "x2")))),
    unname(expected)
  )

  invalid_state = pipeop$clone(deep = TRUE)
  names(invalid_state$state$scalers$block$mean) = c("x1", "wrong")
  expect_error(
    invalid_state$predict(list(predict_task)),
    "missing 1 trained feature name"
  )

  missing_state = pipeop$clone(deep = TRUE)
  missing_state$state$scalers$block = NULL
  expect_error(
    missing_state$predict(list(predict_task)),
    "state is missing"
  )

  constant_task = mlr3::TaskRegr$new(
    "scale_constant",
    data.frame(x1 = 1:4, x2 = 1, y = 1:4),
    target = "y"
  )
  constant_pipeop = PipeOpBlockScaling$new(param_vals = list(
    blocks = list(block = c("x1", "x2")),
    method = "feature_zscore"
  ))
  expect_error(constant_pipeop$train(list(constant_task)),
    "zero-variance")

  missing_task = mlr3::TaskRegr$new(
    "scale_missing",
    data.frame(x1 = c(1, NA, 3), x2 = 1:3, y = 1:3),
    target = "y"
  )
  missing_pipeop = PipeOpBlockScaling$new(param_vals = list(
    blocks = list(block = c("x1", "x2")),
    method = "feature_zscore"
  ))
  expect_error(missing_pipeop$train(list(missing_task)), "finite")
})


test_that("PipeOpBlockScaling rejects implicit blocks without numeric features", {
  task = mlr3::TaskRegr$new("scale_no_numeric", data.frame(
    category = factor(c("A", "B", "A")), y = 1:3
  ), target = "y")
  expect_error(PipeOpBlockScaling$new()$train(list(task)),
    "no numeric features")
})


test_that("block scaling preserves targets colliding with internal key names", {
  task = mlr3::TaskRegr$new("scale_key_target", data.frame(
    x = c(1, 3, 8, 11), `..row_id_blockscale` = c(10, 30, 70, 100)
  ), target = "..row_id_blockscale")
  pipeop = PipeOpBlockScaling$new()
  expect_equal(pipeop$train(list(task))[[1L]]$truth(), task$truth())
  expect_equal(pipeop$predict(list(task))[[1L]]$truth(), task$truth())
})


test_that("block scaling output tasks keep column information consistent", {
  set.seed(56)
  n = 60L
  data = data.frame(
    x = rnorm(n, 5), xi = sample(1:20, n, replace = TRUE), z = sample(1:5, n, replace = TRUE),
    g = rep(seq_len(12L), 5L), y = factor(rep(c("a", "b", "b"), n / 3L))
  )
  task = mlr3::TaskClassif$new("blockscale_colinfo", data, target = "y")
  task$set_col_roles("g", "group")
  task$labels = c(x = "x label")
  pipeop = PipeOpBlockScaling$new(param_vals = list(
    blocks = list(b1 = c("x", "xi")), method = "feature_zscore"
  ))
  for (out in list(pipeop$train(list(task))[[1L]], pipeop$predict(list(task))[[1L]])) {
    expect_true(setequal(out$col_info$id, out$backend$colnames))
    expect_identical(out$feature_names, task$feature_names)
    expect_identical(out$feature_types[id == "xi", type], "numeric")
    expect_identical(out$feature_types[id == "z", type], "integer")
    expect_identical(out$col_roles$group, "g")
    expect_identical(unname(out$labels[["x"]]), "x label")
    expect_equal(out$truth(), task$truth())
    expect_true(is.double(out$data(cols = "xi")$xi))
    expect_silent(out$data(cols = out$backend$primary_key))
  }

  unscaled = PipeOpBlockScaling$new(param_vals = list(
    blocks = list(b1 = c("x", "xi")), method = "none"
  ))$train(list(task))[[1L]]
  expect_identical(unscaled$feature_types[id == "xi", type], "integer")
  expect_equal(unscaled$data(cols = "xi")$xi, data$xi)

  graph = mlr3pipelines::`%>>%`(
    mlr3pipelines::po("blockscale", blocks = list(b1 = c("x", "xi")), method = "unit_ssq"),
    mlr3pipelines::po("scale", affect_columns = mlr3pipelines::selector_type("numeric"))
  )
  scaled = graph$train(task)[[1L]]
  expect_equal(unname(colMeans(scaled$data(cols = c("x", "xi")))), c(0, 0))

  ungrouped = task$clone()
  ungrouped$col_roles$group = character(0)
  balancing = mlr3pipelines::`%>>%`(
    mlr3pipelines::po("blockscale", blocks = list(b1 = c("x", "xi"))),
    mlr3pipelines::po("classbalancing", adjust = "minor", reference = "major")
  )
  balanced = balancing$train(ungrouped)[[1L]]
  expect_equal(as.integer(table(balanced$truth())), c(40L, 40L))
})
