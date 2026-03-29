test_that("MB-sPLS measure scoring uses the payload for the learner run_id", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 1L)
  blocks = task$block_features()
  log_env = new.env(parent = emptyenv())

  po_mb = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      log_env = log_env,
      append = FALSE
    )
  )
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )

  gl$train(task)
  prediction = gl$predict(task)

  run_id = tryCatch(gl$model$mbspls$state$run_id, error = function(e) NULL) %||%
    tryCatch(gl$graph$pipeops$mbspls$state$run_id, error = function(e) NULL)
  if (!(is.character(run_id) && nzchar(run_id))) {
    run_id = "forced_test_run_id"
    gl$model$mbspls$state$run_id = run_id
    try(gl$graph$pipeops$mbspls$state$run_id <- run_id, silent = TRUE)
  }

  payload = list(
    mac_comp = c(0.321),
    ev_comp = c(0.5),
    ev_block = matrix(0.5, nrow = 1L, ncol = length(blocks)),
    perf_metric = "mac",
    blocks = names(blocks)
  )
  log_env$mbspls_last[[run_id]] = payload
  log_env$last = list(
    mac_comp = 0,
    ev_comp = 0,
    ev_block = matrix(0, nrow = 1L, ncol = length(blocks)),
    perf_metric = "mac",
    blocks = names(blocks)
  )
  expected = mbspls_measure_score_from_payload(payload, "mbspls.mac")

  expect_equal(
    mlr3::msr("mbspls.mac")$score(prediction = prediction, task = task, learner = gl),
    expected
  )
})


test_that("MB-sPCA measure scoring uses the payload for the learner run_id", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 2L)
  blocks = task$block_features()
  log_env = new.env(parent = emptyenv())

  po_mb = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      log_env = log_env
    )
  )
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )

  gl$train(task)
  prediction = gl$predict(task)

  run_id = tryCatch(gl$model$mbspca$state$run_id, error = function(e) NULL) %||%
    tryCatch(gl$graph$pipeops$mbspca$state$run_id, error = function(e) NULL)
  if (!(is.character(run_id) && nzchar(run_id))) {
    run_id = "forced_test_run_id"
    gl$model$mbspca$state$run_id = run_id
    try(gl$graph$pipeops$mbspca$state$run_id <- run_id, silent = TRUE)
  }

  payload = list(
    ev_comp = c(0.111, 0.333),
    ev_block = matrix(c(0.1, 0.3), nrow = 1L),
    blocks = names(blocks)
  )
  log_env$mbspls_last[[run_id]] = payload
  log_env$last = list(ev_comp = 0, ev_block = matrix(0, nrow = 1L, ncol = length(blocks)), blocks = names(blocks))
  truth = mbspca_measure_score_from_payload(payload, "mbspca.mean_ev")

  expect_equal(
    mlr3::msr("mbspca.mean_ev")$score(prediction = prediction, task = task, learner = gl),
    truth
  )
})


test_that("node lookup prefers a fitted-node log_env over the template log_env", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 3L)
  blocks = task$block_features()
  env_tpl = new.env(parent = emptyenv())

  po_mb = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      log_env = env_tpl,
      append = FALSE
    )
  )
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )

  gl$train(task)

  env_fit = new.env(parent = emptyenv())
  gl$model[["mbspls"]]$param_set$values$log_env = env_fit

  nodes = .mbspls_locate_nodes_general(gl)
  expect_identical(nodes$fit_env, env_fit)
})
