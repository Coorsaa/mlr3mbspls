make_pw_signal_data = function(n = 80L, seed = 1L) {
  set.seed(seed)
  l1 = rnorm(n)
  l2 = rnorm(n)
  make_block = function(prefix) {
    m = cbind(
      l1 + rnorm(n, sd = 0.4),
      l2 + rnorm(n, sd = 0.6),
      matrix(rnorm(n * 4L), n)
    )
    colnames(m) = paste0(prefix, seq_len(ncol(m)))
    m
  }
  b1 = make_block("x")
  b2 = make_block("z")
  list(
    task = mlr3::TaskRegr$new(
      id = "pw_consistency",
      backend = data.frame(b1, b2, y = l1 + rnorm(n)),
      target = "y"
    ),
    blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  )
}

lv_matrix = function(task) {
  cols = sort(grep("^LV", task$feature_names, value = TRUE))
  as.matrix(task$data(cols = cols))
}


test_that("stability-only selection keeps emitted LV features identical at train and predict", {
  d = make_pw_signal_data()
  log_env = new.env(parent = emptyenv())
  graph = po("mbspls",
    blocks = d$blocks, ncomp = 2L, c_b1 = 1.5, c_b2 = 1.5,
    store_train_blocks = TRUE, log_env = log_env
  ) %>>%
    po("mbspls_bootstrap_select",
      stability_only = TRUE, B = 20L, seed_bootstrap = 1L, log_env = log_env
    )

  out_train = graph$train(d$task)[[1L]]
  out_pred = graph$predict(d$task)[[1L]]

  # The default predict_weights = "auto" must not switch to stable weights:
  # the learner downstream was trained on the raw-weight LVs.
  expect_equal(lv_matrix(out_pred), lv_matrix(out_train), tolerance = 1e-10)
  expect_identical(log_env$last$weights_source, "raw")
  expect_identical(log_env$last$emitted_weights_source, "raw")

  graph$pipeops$mbspls$param_set$values$predict_weights = "stable_ci"
  expect_error(graph$predict(d$task), "stability_only = TRUE")
})


test_that("stable prediction weights change the payload but never the emitted LV features", {
  d = make_pw_signal_data(seed = 2L)
  log_env = new.env(parent = emptyenv())
  pipeop = PipeOpMBsPLS$new(
    blocks = d$blocks,
    param_vals = list(
      ncomp = 2L, c_b1 = 1.5, c_b2 = 1.5,
      log_env = log_env, predict_weights = "stable_ci"
    )
  )
  pipeop$train(list(d$task))
  st = pipeop$state

  # Publish stable weights for this run as a bootstrap-selection stage would.
  W_stable = lapply(st$weights, function(wk) {
    lapply(wk, function(wb) {
      wb[abs(wb) < max(abs(wb))] = 0
      wb
    })
  })
  st_env = log_env$mbspls_states[[st$run_id]]
  st_env$weights_stable_ci = W_stable
  st_env$loadings_stable_ci = st$loadings
  st_env$selection_method = "ci"
  log_env_store_state(log_env, st_env, warn_overwrite = FALSE)

  out = pipeop$predict(list(d$task))[[1L]]

  expect_equal(lv_matrix(out), st$T_mat[, sort(colnames(st$T_mat))], tolerance = 1e-10)
  expect_identical(log_env$last$weights_source, "stable_ci")
  expect_identical(log_env$last$emitted_weights_source, "raw")
  expect_equal(log_env$last$weights, W_stable, ignore_attr = TRUE)
  expect_false(isTRUE(all.equal(log_env$last$T_mat, st$T_mat, check.attributes = FALSE)))
})


test_that("the bootstrap-selection graph emits identical stable LVs at train and predict", {
  d = make_pw_signal_data(seed = 3L)
  # Frequency selection keeps the dominant signal feature; the strict CI rule
  # can legitimately keep nothing for one-feature components and few replicates.
  graph = mbspls_graph(
    blocks = d$blocks, ncomp = 1L, B = 20L, seed_bootstrap = 1L,
    selection_method = "frequency"
  )

  out_train = graph$train(d$task)[[1L]]
  out_pred = graph$predict(d$task)[[1L]]

  expect_equal(lv_matrix(out_pred), lv_matrix(out_train), tolerance = 1e-10)
})
