test_that("LearnerClassifKNNGower - basic train/predict with mixed-type features", {
  set.seed(1)
  n = 60
  df = data.frame(
    x1  = rnorm(n),
    x2  = rnorm(n),
    cat = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    ord = ordered(sample(1:3, n, replace = TRUE)),
    y   = factor(sample(c("neg", "pos"), n, replace = TRUE))
  )
  task = mlr3::TaskClassif$new(id = "knn_basic", backend = df, target = "y")
  lrn = mlr3::lrn("classif.knngower", k = 5L)
  lrn$train(task)
  pred = lrn$predict(task)
  expect_s3_class(pred, "PredictionClassif")
  expect_equal(length(pred$response), n)
})


test_that("LearnerClassifKNNGower - predicts when a factor level is absent from training batch", {
  set.seed(2)
  n_train = 50
  # Level "c" is in the level set but ALL training observations are "a" or "b"
  df_train = data.frame(
    x1  = rnorm(n_train),
    cat = factor(sample(c("a", "b"), n_train, replace = TRUE), levels = c("a", "b", "c")),
    y   = factor(sample(c("neg", "pos"), n_train, replace = TRUE))
  )
  # Predict data contains "c" (known level, but never seen in training)
  df_test = data.frame(
    x1  = rnorm(20),
    cat = factor(c(rep("a", 10), rep("c", 10)), levels = c("a", "b", "c")),
    y   = factor(c(rep("neg", 10), rep("pos", 10))) # both levels must be present
  )
  task_train = mlr3::TaskClassif$new(id = "knn_rare", backend = df_train, target = "y")
  task_test = mlr3::TaskClassif$new(id = "knn_rare_te", backend = df_test, target = "y")

  lrn = mlr3::lrn("classif.knngower", k = 3L, predict_type = "prob")
  lrn$train(task_train)
  # Should NOT error; "c" is a known level, so it gets a valid positive code
  pred = lrn$predict(task_test)
  expect_s3_class(pred, "PredictionClassif")
  expect_equal(nrow(pred$prob), 20L)
})


test_that("LearnerClassifKNNGower - inverse weighting handles near-duplicate rows", {
  set.seed(3)
  n = 40
  # Create near-duplicate rows to exercise the small-distance path
  x_base = rnorm(n)
  df = data.frame(
    x1 = c(x_base, x_base + 1e-14), # near-identical pairs
    y  = factor(sample(c("neg", "pos"), 2 * n, replace = TRUE))
  )
  task = mlr3::TaskClassif$new(id = "knn_dup", backend = df, target = "y")
  lrn = mlr3::lrn("classif.knngower", k = 3L, weights = "inverse")
  lrn$train(task)
  pred = lrn$predict(task)
  expect_s3_class(pred, "PredictionClassif")
  # No non-finite probabilities
  expect_true(all(is.finite(pred$prob)))
})


test_that("LearnerRegrKNNGower - basic train/predict with mixed-type features", {
  set.seed(4)
  n = 60
  df = data.frame(
    x1  = rnorm(n),
    x2  = rnorm(n),
    cat = factor(sample(c("a", "b"), n, replace = TRUE)),
    y   = rnorm(n)
  )
  task = mlr3::TaskRegr$new(id = "knn_regr", backend = df, target = "y")
  lrn = mlr3::lrn("regr.knngower", k = 5L)
  lrn$train(task)
  pred = lrn$predict(task)
  expect_s3_class(pred, "PredictionRegr")
  expect_equal(length(pred$response), n)
  expect_true(all(is.finite(pred$response)))
})


test_that("LearnerRegrKNNGower - predicts when a factor level is absent from training batch", {
  set.seed(5)
  n_train = 40
  df_train = data.frame(
    x1  = rnorm(n_train),
    cat = factor(sample(c("a", "b"), n_train, replace = TRUE), levels = c("a", "b", "z")),
    y   = rnorm(n_train)
  )
  # predict batch includes "z" (known level, never seen in training)
  df_test = data.frame(
    x1  = rnorm(10),
    cat = factor(c(rep("b", 5), rep("z", 5)), levels = c("a", "b", "z")),
    y   = rnorm(10)
  )
  task_train = mlr3::TaskRegr$new(id = "knn_ru_tr", backend = df_train, target = "y")
  task_test = mlr3::TaskRegr$new(id = "knn_ru_te", backend = df_test, target = "y")

  lrn = mlr3::lrn("regr.knngower", k = 3L, predict_type = "se")
  lrn$train(task_train)
  pred = lrn$predict(task_test)
  expect_s3_class(pred, "PredictionRegr")
  expect_true(all(is.finite(pred$response)))
})


test_that("LearnerRegrKNNGower - inverse weighting handles near-duplicate rows", {
  set.seed(6)
  n = 30
  x_base = rnorm(n)
  df = data.frame(
    x1 = c(x_base, x_base + 1e-14),
    y  = c(rnorm(n), rnorm(n))
  )
  task = mlr3::TaskRegr$new(id = "knn_regr_dup", backend = df, target = "y")
  lrn = mlr3::lrn("regr.knngower", k = 3L, weights = "inverse")
  lrn$train(task)
  pred = lrn$predict(task)
  expect_true(all(is.finite(pred$response)))
})


test_that("LearnerClassifKNNGower - all-NA feature column handled via min_feature_frac", {
  set.seed(7)
  n = 40
  df_train = data.frame(
    x1  = rnorm(n),
    x2  = NA_real_, # all-NA column
    y   = factor(sample(c("a", "b"), n, replace = TRUE))
  )
  df_test = data.frame(
    x1  = rnorm(10),
    x2  = NA_real_,
    y   = factor(sample(c("a", "b"), 10, replace = TRUE))
  )
  task_train = mlr3::TaskClassif$new(id = "knn_na", backend = df_train, target = "y")
  task_test = mlr3::TaskClassif$new(id = "knn_na_te", backend = df_test, target = "y")

  # Set min_feature_frac low enough to still predict with only x1
  lrn = mlr3::lrn("classif.knngower", k = 3L, min_feature_frac = 0.4)
  lrn$train(task_train)
  pred = lrn$predict(task_test)
  expect_s3_class(pred, "PredictionClassif")
})


test_that("regression Gower prediction honors fail-on-missing policy", {
  task = mlr3::TaskRegr$new("knn_complete", data.frame(x = 1:5, y = 1:5),
    target = "y")
  learner = LearnerRegrKNNGower$new()
  learner$param_set$set_values(k = 1L, na_handling = "fail")
  learner$train(task)
  expect_error(learner$predict_newdata(data.frame(x = NA_real_)),
    "Prediction data contain missing features")
})


test_that("singleton ordered levels do not turn missing values into matches", {
  task = mlr3::TaskRegr$new("knn_ordered_missing", data.frame(
    ord = ordered(c("only", NA), levels = "only"), y = c(10, 100)
  ), target = "y")
  learner = LearnerRegrKNNGower$new()
  learner$param_set$set_values(k = 2L, weights = "uniform")
  learner$train(task)
  prediction = learner$predict_newdata(data.frame(
    ord = ordered("only", levels = "only")
  ))
  expect_equal(prediction$response, 10)
  expect_error(learner$predict_newdata(data.frame(
    ord = ordered(NA_character_, levels = "only")
  )), "no (eligible |valid )?neighbou?rs|No (eligible |valid )?neighbou?rs")
})


test_that("classification Gower prediction honors fail-on-missing policy", {
  task = mlr3::TaskClassif$new("knn_classif_complete", data.frame(
    x = 1:6, y = factor(rep(c("A", "B"), 3L))
  ), target = "y")
  learner = LearnerClassifKNNGower$new()
  learner$param_set$set_values(k = 1L, na_handling = "fail")
  learner$train(task)
  expect_error(learner$predict_newdata(data.frame(x = NA_real_)),
    "Prediction data contain missing features")
  expect_s3_class(learner$predict_newdata(data.frame(x = 2)), "PredictionClassif")
})


test_that("classification: singleton ordered levels do not turn missing values into matches", {
  task = mlr3::TaskClassif$new("knn_classif_ordered_missing", data.frame(
    ord = ordered(c("only", NA), levels = "only"), y = factor(c("A", "B"))
  ), target = "y")
  learner = LearnerClassifKNNGower$new()
  learner$param_set$set_values(k = 2L, weights = "uniform")
  learner$predict_type = "prob"
  learner$train(task)
  prediction = learner$predict_newdata(data.frame(
    ord = ordered("only", levels = "only")
  ))
  expect_equal(unname(prediction$prob[1L, "A"]), 1)
  expect_error(learner$predict_newdata(data.frame(
    ord = ordered(NA_character_, levels = "only")
  )), "no (eligible |valid )?neighbou?rs|No (eligible |valid )?neighbou?rs")
})


test_that("classification and regression share the Gower encoding of missing ordered values", {
  data = data.frame(
    x = c(6.5, 4, 0, 10),
    ord = ordered(c("only", NA, "only", "only"), levels = "only")
  )
  query = data.frame(x = 5, ord = ordered("only", levels = "only"))

  classif = LearnerClassifKNNGower$new()
  classif$param_set$set_values(k = 1L)
  classif$train(mlr3::TaskClassif$new("knn_shared_classif",
    cbind(data, y = factor(c("A", "B", "B", "B"))), target = "y"))
  expect_identical(as.character(classif$predict_newdata(query)$response), "A")

  regr = LearnerRegrKNNGower$new()
  regr$param_set$set_values(k = 1L)
  regr$train(mlr3::TaskRegr$new("knn_shared_regr", cbind(data, y = c(1, 2, 3, 4)),
    target = "y"))
  expect_equal(regr$predict_newdata(query)$response, 1)
  expect_identical(classif$model$Xord, regr$model$Xord)

  encoded = mlr3mbspls:::knn_gower_encode_blocks(
    data.table::data.table(ord = factor(c("only", "new", NA))),
    num_cols = character(0), cat_cols = character(0), ord_cols = "ord",
    ref = list(ranges_num = numeric(0), cat_levels = list(), ord_levels = list("only"))
  )
  expect_equal(encoded$Xord[, 1L], c(0, NA, NA))
})


test_that("Gower encodings are fitted on training data and frozen at prediction", {
  set.seed(8)
  train = data.frame(
    x = runif(30), cat = factor(sample(c("a", "b"), 30, replace = TRUE)),
    ord = ordered(sample(c("lo", "mid", "hi"), 30, replace = TRUE), levels = c("lo", "mid", "hi"))
  )
  query = data.frame(
    x = c(0.2, 50, -20),
    cat = factor(c("a", "b", "a"), levels = c("a", "b")),
    ord = ordered(c("mid", "hi", "lo"), levels = c("lo", "mid", "hi"))
  )
  learners = list(
    regr = list(LearnerRegrKNNGower$new(), function(data) {
      mlr3::TaskRegr$new("knn_frozen_regr", cbind(data, y = rnorm(30)), target = "y")
    }),
    classif = list(LearnerClassifKNNGower$new(), function(data) {
      mlr3::TaskClassif$new("knn_frozen_classif",
        cbind(data, y = factor(sample(c("A", "B"), 30, replace = TRUE))), target = "y")
    })
  )
  for (entry in learners) {
    learner = entry[[1L]]
    learner$param_set$set_values(k = 3L)
    if ("prob" %in% learner$predict_types) learner$predict_type = "prob"
    learner$train(entry[[2L]](train))
    model = learner$model
    batch = learner$predict_newdata(query)
    single = learner$predict_newdata(query[1L, , drop = FALSE])
    if (inherits(batch, "PredictionRegr")) {
      expect_equal(single$response, batch$response[[1L]])
    } else {
      expect_equal(single$prob[1L, ], batch$prob[1L, ])
    }
    expect_identical(learner$model, model)
  }
})
