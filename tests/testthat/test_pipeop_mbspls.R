test_that("PipeOpMBsPLS - trains and predicts with per-block c parameters", {
  set.seed(42)

  n = 80
  p_clin = 6
  p_gen = 20

  clinical = matrix(rnorm(n * p_clin), nrow = n, ncol = p_clin)
  colnames(clinical) = paste0("c", seq_len(p_clin))
  genomic = matrix(rnorm(n * p_gen), nrow = n, ncol = p_gen)
  colnames(genomic) = paste0("g", seq_len(p_gen))
  age = rnorm(n)
  y = rnorm(n)

  # Add a non-block feature (age) to verify append=TRUE behaviour.
  df = data.frame(clinical, genomic, age = age, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(
    clinical = colnames(clinical),
    genomic = colnames(genomic)
  )

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 2L,
      c_clinical = sqrt(5L),
      c_genomic = sqrt(6L),
      append = FALSE
    )
  )

  out_train = po$train(list(task))[[1]]
  lv_cols = c(
    "LV1_clinical", "LV1_genomic",
    "LV2_clinical", "LV2_genomic"
  )

  expect_s3_class(out_train, "Task")
  expect_setequal(out_train$feature_names, lv_cols)

  st = po$state
  expect_true(is.list(st))
  expect_equal(st$ncomp, 2L)
  expect_setequal(names(st$blocks), names(blocks))
  expect_length(st$weights, 2L)
  expect_setequal(names(st$weights[[1]]), names(blocks))
  expect_true(all(vapply(st$weights[[1]], is.numeric, logical(1))))

  task_new = task$clone(deep = TRUE)
  task_new$filter(1:10)

  out_pred = po$predict(list(task_new))[[1]]
  expect_s3_class(out_pred, "Task")
  expect_equal(out_pred$nrow, 10)
  expect_setequal(out_pred$feature_names, lv_cols)
})


test_that("PipeOpMBsPLS - append = TRUE keeps raw features and adds LVs", {
  set.seed(123)

  n = 60
  p_clin = 4
  p_gen = 10

  clinical = matrix(rnorm(n * p_clin), nrow = n, ncol = p_clin)
  colnames(clinical) = paste0("c", seq_len(p_clin))
  genomic = matrix(rnorm(n * p_gen), nrow = n, ncol = p_gen)
  colnames(genomic) = paste0("g", seq_len(p_gen))
  age = rnorm(n)
  y = rnorm(n)

  df = data.frame(clinical, genomic, age = age, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(
    clinical = colnames(clinical),
    genomic = colnames(genomic)
  )

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_clinical = 2L,
      c_genomic = 2L,
      append = TRUE
    )
  )

  out_train = po$train(list(task))[[1]]
  lv_cols = c("LV1_clinical", "LV1_genomic")

  expect_true(all(lv_cols %in% out_train$feature_names))
  expect_true("age" %in% out_train$feature_names)
  expect_true(all(c(blocks$clinical, blocks$genomic) %in% out_train$feature_names))
})


test_that("PipeOpMBsPLS - c_matrix path works and validates dimensions", {
  set.seed(1)

  n = 50
  p1 = 5
  p2 = 7

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  Cmat = matrix(sqrt(3), nrow = 2, ncol = 2)
  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 2L,
      c_matrix = Cmat,
      append = FALSE
    )
  )
  out_train = po$train(list(task))[[1]]

  expect_true(is.matrix(po$state$c_matrix))
  expect_equal(dim(po$state$c_matrix), c(2L, 2L))
  expect_true(all(c("LV1_b1", "LV1_b2", "LV2_b1", "LV2_b2") %in% out_train$feature_names))

  # wrong dimensions should error early
  expect_error(
    PipeOpMBsPLS$new(
      blocks = blocks,
      param_vals = list(
        ncomp = 2L,
        c_matrix = matrix(1, nrow = 1, ncol = 2),
        append = FALSE
      )
    ),
    sprintf("c_matrix must have %s rows \\(blocks\\); got 1", length(blocks))
  )
})


test_that("PipeOpMBsPLS - writes expected payloads to log_env", {
  set.seed(7)

  n = 40
  p1 = 4
  p2 = 6

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())
  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      store_train_blocks = TRUE,
      append = FALSE
    )
  )

  po$train(list(task))
  expect_true(is.list(log_env$mbspls_state))
  expect_true(all(c("blocks", "weights", "loadings", "T_mat_train") %in% names(log_env$mbspls_state)))

  po$predict(list(task))
  expect_true(is.list(log_env$last))
  expect_true(all(c("mac_comp", "ev_block", "ev_comp", "T_mat") %in% names(log_env$last)))
})
