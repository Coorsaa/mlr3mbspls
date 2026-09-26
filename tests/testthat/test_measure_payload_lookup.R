new_payload_env = function() {
  log_env = new.env(parent = emptyenv())
  log_env$warn_overwrite = FALSE
  log_env
}

mbspls_payload_learner = function(blocks, log_env, c_block = NULL) {
  param_vals = list(ncomp = 1L, log_env = log_env, append = FALSE)
  if (!is.null(c_block)) {
    param_vals[[paste0("c_", names(blocks)[[1L]])]] = c_block
  }
  po_mb = PipeOpMBsPLS$new(blocks = blocks, param_vals = param_vals)
  mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )
}

mbspca_payload_learner = function(blocks, log_env) {
  po_mb = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L, log_env = log_env))
  mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )
}

decoy_payload = function(blocks) {
  list(
    mac_comp = 0,
    ev_comp = 0,
    ev_block = matrix(0, nrow = 1L, ncol = length(blocks)),
    perf_metric = "mac",
    blocks = names(blocks)
  )
}


test_that("MB-sPLS measure scoring uses the payload for the learner run_id", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 1L)
  blocks = task$block_features()
  log_env = new_payload_env()
  gl = mbspls_payload_learner(blocks, log_env)

  gl$train(task)
  prediction = gl$predict(task)

  run_id = gl$model$mbspls[["run_id"]]
  expect_true(is.character(run_id) && length(run_id) == 1L && nzchar(run_id))
  expect_true(is.list(log_env$mbspls_last[[run_id]]))

  payload = list(
    mac_comp = c(0.321),
    ev_comp = c(0.5),
    ev_block = matrix(0.5, nrow = 1L, ncol = length(blocks)),
    perf_metric = "mac",
    blocks = names(blocks)
  )
  log_env$mbspls_last[[run_id]] = payload
  log_env$last = decoy_payload(blocks)

  expect_equal(
    mlr3::msr("mbspls.mac")$score(prediction = prediction, task = task, learner = gl),
    mbspls_measure_score_from_payload(payload, "mbspls.mac")
  )
})


test_that("MB-sPCA measure scoring uses the payload for the learner run_id", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 2L)
  blocks = task$block_features()
  log_env = new_payload_env()
  gl = mbspca_payload_learner(blocks, log_env)

  gl$train(task)
  prediction = gl$predict(task)

  run_id = gl$model$mbspca[["run_id"]]
  expect_true(is.character(run_id) && length(run_id) == 1L && nzchar(run_id))
  expect_true(is.list(log_env$mbspls_last[[run_id]]))

  payload = list(
    ev_comp = c(0.111, 0.333),
    ev_block = matrix(c(0.1, 0.3), nrow = 1L),
    blocks = names(blocks)
  )
  log_env$mbspls_last[[run_id]] = payload
  log_env$last = list(ev_comp = 0, ev_block = matrix(0, nrow = 1L, ncol = length(blocks)), blocks = names(blocks))

  expect_equal(
    mlr3::msr("mbspca.mean_ev")$score(prediction = prediction, task = task, learner = gl),
    mbspca_measure_score_from_payload(payload, "mbspca.mean_ev")
  )
})


test_that("a known run_id without a stored payload scores NA instead of the last payload", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 4L)
  blocks = task$block_features()
  log_env = new_payload_env()
  gl = mbspls_payload_learner(blocks, log_env)
  gl$train(task)
  prediction = gl$predict(task)

  log_env$mbspls_last[[gl$model$mbspls$run_id]] = NULL
  log_env$last = decoy_payload(blocks)

  expect_true(is.na(mlr3::msr("mbspls.mac")$score(prediction = prediction, task = task, learner = gl)))
})


test_that("fitted states without a run_id fall back to the last payload with a warning", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 5L)
  blocks = task$block_features()
  log_env = new_payload_env()
  gl = mbspls_payload_learner(blocks, log_env)
  gl$train(task)
  prediction = gl$predict(task)

  gl$model$mbspls[["run_id"]] = NULL
  log_env$last = decoy_payload(blocks)

  measure = mlr3::msr("mbspls.mac")
  expect_warning(
    measure$score(prediction = prediction, task = task, learner = gl),
    "records no run id"
  )
  score = suppressWarnings(measure$score(prediction = prediction, task = task, learner = gl))
  expect_equal(score, 0)
})


test_that("a learner keeps its own score after another learner predicts on the shared log_env", {
  task = task_multiblock_synthetic(task_type = "clust", n = 40L, seed = 6L)
  blocks = task$block_features()
  log_env = new_payload_env()
  gl_a = mbspls_payload_learner(blocks, log_env, c_block = 1)
  gl_b = mbspls_payload_learner(blocks, log_env, c_block = sqrt(length(blocks[[1L]])))

  gl_a$train(task)
  pred_a = gl_a$predict(task)
  expected_a = mbspls_measure_score_from_payload(log_env$last, "mbspls.mac")

  gl_b$train(task)
  gl_b$predict(task)
  expect_false(identical(log_env$last$run_id, gl_a$model$mbspls$run_id))

  expect_equal(
    mlr3::msr("mbspls.mac")$score(prediction = pred_a, task = task, learner = gl_a),
    expected_a
  )
})


test_that("resampled MB-sPLS scores use each iteration's own payload", {
  task = task_multiblock_synthetic(task_type = "clust", n = 60L, seed = 3L)
  blocks = task$block_features()
  log_env = new_payload_env()
  gl = mbspls_payload_learner(blocks, log_env)

  rr = mlr3::resample(task, gl, mlr3::rsmp("cv", folds = 3L), store_models = TRUE)
  expect_length(log_env$mbspls_last, 3L)

  for (id in c("mbspls.mac", "mbspls.mac_evwt", "mbspls.ev")) {
    scores_dt = rr$score(mlr3::msr(id))
    run_ids = vapply(scores_dt$learner, function(l) l$model$mbspls$run_id, character(1L))
    expect_length(unique(run_ids), 3L)
    own = vapply(run_ids, function(run_id) {
      mbspls_measure_score_from_payload(log_env$mbspls_last[[run_id]], id)
    }, numeric(1L), USE.NAMES = FALSE)
    all_logged = vapply(log_env$mbspls_last, mbspls_measure_score_from_payload, numeric(1L), measure = id)

    expect_equal(scores_dt[[id]], own)
    expect_equal(sort(scores_dt[[id]]), sort(unname(all_logged)))
    expect_gt(length(unique(round(scores_dt[[id]], 12L))), 1L)
    expect_equal(unname(rr$aggregate(mlr3::msr(id))), mean(own))
  }
})


test_that("resampled MB-sPCA scores use each iteration's own payload", {
  task = task_multiblock_synthetic(task_type = "clust", n = 60L, seed = 8L)
  blocks = task$block_features()
  log_env = new_payload_env()
  gl = mbspca_payload_learner(blocks, log_env)

  rr = mlr3::resample(task, gl, mlr3::rsmp("cv", folds = 3L), store_models = TRUE)
  scores_dt = rr$score(mlr3::msr("mbspca.mean_ev"))
  run_ids = vapply(scores_dt$learner, function(l) l$model$mbspca$run_id, character(1L))
  expect_length(unique(run_ids), 3L)
  own = vapply(run_ids, function(run_id) {
    mbspca_measure_score_from_payload(log_env$mbspls_last[[run_id]], "mbspca.mean_ev")
  }, numeric(1L), USE.NAMES = FALSE)

  expect_equal(scores_dt$mbspca.mean_ev, own)
  expect_gt(length(unique(round(own, 12L))), 1L)
})


test_that("MB-sPLS and MB-sPCA measures require stored models", {
  for (id in c("mbspls.mac_evwt", "mbspls.mac", "mbspls.ev", "mbspls.block_ev", "mbspca.mean_ev")) {
    expect_true("requires_model" %in% mlr3::msr(id)$properties)
  }

  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 9L)
  gl = mbspls_payload_learner(task$block_features(), new_payload_env())
  rr = mlr3::resample(task, gl, mlr3::rsmp("cv", folds = 2L), store_models = FALSE)
  expect_error(rr$score(mlr3::msr("mbspls.mac")), "requires the trained model")
})


test_that("node lookup resolves the fitted state and the template log_env", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 3L)
  blocks = task$block_features()
  env_tpl = new_payload_env()
  gl = mbspls_payload_learner(blocks, env_tpl)

  gl$train(task)

  nodes = .mbspls_locate_nodes_general(gl)
  expect_identical(nodes$fit_env, env_tpl)
  expect_identical(nodes$fit_state$run_id, gl$model$mbspls$run_id)
})
