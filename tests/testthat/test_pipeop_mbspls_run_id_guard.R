test_that("PipeOpMBsPLS ignores shared log_env stable weights when run_id is missing", {
  set.seed(1234)

  n = 36L
  b1 = matrix(rnorm(n * 4L), nrow = n)
  colnames(b1) = paste0("x", seq_len(ncol(b1)))
  b2 = matrix(rnorm(n * 5L), nrow = n)
  colnames(b2) = paste0("z", seq_len(ncol(b2)))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_runid_guard", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      append = FALSE
    )
  )
  po$train(list(task))

  task_new = task$clone(deep = TRUE)
  task_new$filter(1:8)
  raw_pred = po$predict(list(task_new$clone(deep = TRUE)))[[1L]]

  zero_weights = lapply(po$state$weights, function(wk) {
    lapply(wk, function(wb) stats::setNames(rep(0, length(wb)), names(wb)))
  })

  bogus_env = new.env(parent = emptyenv())
  bogus_state = list(
    run_id = "bogus_run",
    blocks = po$state$blocks,
    weights = po$state$weights,
    weights_stable = zero_weights,
    selection_method = "ci"
  )
  bogus_env$mbspls_state = bogus_state
  bogus_env$mbspls_states = list(bogus_run = bogus_state)
  bogus_env$mbspls_state_last_id = "bogus_run"

  po_env = po$clone(deep = TRUE)
  po_env$param_set$values$log_env = bogus_env
  guarded_pred = po_env$predict(list(task_new$clone(deep = TRUE)))[[1L]]

  expect_equal(
    as.matrix(guarded_pred$data(cols = guarded_pred$feature_names)),
    as.matrix(raw_pred$data(cols = raw_pred$feature_names)),
    tolerance = 1e-8
  )
})
