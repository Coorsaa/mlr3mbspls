make_offset_data = function(n = 70L, seed = 61L) {
  set.seed(seed)
  l1 = rnorm(n)
  l2 = rnorm(n)
  make_block = function(prefix) {
    m = cbind(
      l1 + rnorm(n, sd = 0.4),
      l2 + rnorm(n, sd = 0.5),
      matrix(rnorm(n * 3L), n)
    )
    colnames(m) = paste0(prefix, seq_len(ncol(m)))
    m
  }
  x = cbind(make_block("a"), make_block("b"))
  offsets = stats::setNames(seq(-20, 25, length.out = ncol(x)), colnames(x))
  list(
    x = x,
    offsets = offsets,
    y = l1 + rnorm(n),
    blocks = list(a = paste0("a", 1:5), b = paste0("b", 1:5))
  )
}

shifted = function(x, offsets) sweep(x, 2L, offsets[colnames(x)], "+")


test_that("PipeOpMBsPLS is invariant to column shifts of the training and prediction data", {
  d = make_offset_data()
  train_rows = 1:50
  test_rows = 51:70
  fit_on = function(x) {
    task = mlr3::TaskRegr$new("mbspls_shift", data.frame(x, y = d$y), target = "y")
    pipeop = PipeOpMBsPLS$new(
      blocks = d$blocks,
      param_vals = list(ncomp = 2L, c_a = 1.6, c_b = 1.6)
    )
    task_train = task$clone(deep = TRUE)$filter(train_rows)
    task_test = task$clone(deep = TRUE)$filter(test_rows)
    pipeop$train(list(task_train))
    pred = pipeop$predict(list(task_test))[[1L]]
    list(pipeop = pipeop, pred = as.matrix(pred$data(cols = pred$feature_names)))
  }
  plain = fit_on(d$x)
  moved = fit_on(shifted(d$x, d$offsets))

  expect_equal(moved$pipeop$state$weights, plain$pipeop$state$weights, tolerance = 1e-8)
  expect_equal(moved$pipeop$state$loadings, plain$pipeop$state$loadings, tolerance = 1e-8)
  expect_equal(moved$pipeop$state$ev_block, plain$pipeop$state$ev_block, tolerance = 1e-8)
  expect_equal(moved$pipeop$state$ev_comp, plain$pipeop$state$ev_comp, tolerance = 1e-8)
  expect_equal(moved$pipeop$state$T_mat, plain$pipeop$state$T_mat, tolerance = 1e-8)
  expect_equal(moved$pred, plain$pred, tolerance = 1e-8)
  expect_equal(
    moved$pipeop$state$center$a,
    colMeans(shifted(d$x, d$offsets)[train_rows, d$blocks$a])
  )
})


test_that("PipeOpMBsPLS stores centred training blocks and the centre in log_env", {
  d = make_offset_data(seed = 62L)
  log_env = new.env(parent = emptyenv())
  task = mlr3::TaskRegr$new("mbspls_centre_env", data.frame(shifted(d$x, d$offsets), y = d$y), target = "y")
  pipeop = PipeOpMBsPLS$new(
    blocks = d$blocks,
    param_vals = list(ncomp = 1L, log_env = log_env, store_train_blocks = TRUE)
  )
  pipeop$train(list(task))
  snapshot = log_env$mbspls_states[[pipeop$state$run_id]]

  expect_identical(snapshot$center, pipeop$state$center)
  expect_equal(unname(colMeans(snapshot$X_train_blocks$a)), rep(0, 5L), tolerance = 1e-10)

  # compute_pipeop_test_ev() applies the same training centre.
  X_raw = lapply(pipeop$state$blocks, function(cols) as.matrix(task$data(cols = cols)))
  ev = compute_pipeop_test_ev(X_raw, pipeop$state)
  expect_equal(unname(ev$ev_comp), unname(pipeop$state$ev_comp), tolerance = 1e-8)
})


test_that("PipeOpMBsPLS predicts with states fitted before centring was stored", {
  d = make_offset_data(seed = 63L)
  task = mlr3::TaskRegr$new("mbspls_old_state", data.frame(d$x, y = d$y), target = "y")
  pipeop = PipeOpMBsPLS$new(blocks = d$blocks, param_vals = list(ncomp = 1L))
  pipeop$train(list(task))
  old_state = pipeop$state
  old_state$center = NULL
  pipeop$state = old_state
  expect_no_error(pipeop$predict(list(task)))
})


test_that("PipeOpMBsPLSXY centres the X blocks with training means", {
  d = make_offset_data(seed = 64L)
  fit_on = function(x) {
    task = mlr3::TaskRegr$new("mbxy_shift", data.frame(x, y = d$y), target = "y")
    pipeop = PipeOpMBsPLSXY$new(blocks = d$blocks, param_vals = list(ncomp = 1L))
    pipeop$train(list(task$clone(deep = TRUE)$filter(1:50)))
    pred = pipeop$predict(list(task$clone(deep = TRUE)$filter(51:70)))[[1L]]
    list(pipeop = pipeop, pred = as.matrix(pred$data(cols = pred$feature_names)))
  }
  plain = fit_on(d$x)
  moved = fit_on(shifted(d$x, d$offsets))

  expect_equal(moved$pipeop$state$weights_x, plain$pipeop$state$weights_x, tolerance = 1e-8)
  expect_equal(moved$pipeop$state$loadings_x, plain$pipeop$state$loadings_x, tolerance = 1e-8)
  expect_equal(moved$pred, plain$pred, tolerance = 1e-8)
  expect_named(moved$pipeop$state$center, names(d$blocks))
})


test_that("PipeOpMBsPCA is invariant to column shifts of the training and prediction data", {
  d = make_offset_data(seed = 65L)
  fit_on = function(x) {
    task = mlr3::TaskUnsupervised$new("mbspca_shift", backend = data.frame(x))
    pipeop = PipeOpMBsPCA$new(blocks = d$blocks, param_vals = list(ncomp = 2L))
    pipeop$train(list(task$clone(deep = TRUE)$filter(1:50)))
    pred = pipeop$predict(list(task$clone(deep = TRUE)$filter(51:70)))[[1L]]
    list(pipeop = pipeop, pred = as.matrix(pred$data(cols = pred$feature_names)))
  }
  plain = fit_on(d$x)
  moved = fit_on(shifted(d$x, d$offsets))

  expect_equal(moved$pipeop$state$weights, plain$pipeop$state$weights, tolerance = 1e-8)
  expect_equal(moved$pipeop$state$ev_block, plain$pipeop$state$ev_block, tolerance = 1e-8)
  expect_equal(moved$pred, plain$pred, tolerance = 1e-8)
})
