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
  pred = gl$predict(task)

  run_id = gl$model$mbspls$run_id
  if (is.null(run_id) || !nzchar(as.character(run_id))) {
    run_id = gl$graph$pipeops$mbspls$state$run_id
  }
  expect_true(is.character(run_id) && nzchar(run_id))
  expect_true(is.list(log_env$mbspls_last[[run_id]]))

  truth = mbspls_measure_score_from_payload(log_env$mbspls_last[[run_id]], "mbspls.mac")
  log_env$last = list(
    mac_comp = 0,
    ev_comp = 0,
    ev_block = matrix(0, nrow = 1L, ncol = length(blocks)),
    perf_metric = "mac",
    blocks = names(blocks)
  )

  expect_equal(
    mlr3::msr("mbspls.mac")$score(prediction = pred, task = task, learner = gl),
    truth
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
  pred = gl$predict(task)

  run_id = gl$model$mbspca$run_id
  if (is.null(run_id) || !nzchar(as.character(run_id))) {
    run_id = gl$graph$pipeops$mbspca$state$run_id
  }
  expect_true(is.character(run_id) && nzchar(run_id))
  expect_true(is.list(log_env$mbspls_last[[run_id]]))

  truth = mbspca_measure_score_from_payload(log_env$mbspls_last[[run_id]], "mbspca.mean_ev")
  log_env$last = list(ev_comp = 0, ev_block = matrix(0, nrow = 1L, ncol = length(blocks)), blocks = names(blocks))

  expect_equal(
    mlr3::msr("mbspca.mean_ev")$score(prediction = pred, task = task, learner = gl),
    truth
  )
})
