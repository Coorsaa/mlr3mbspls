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
