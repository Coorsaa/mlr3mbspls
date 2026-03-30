test_that("TunerSeqMBsPLS rejects unsupported measures", {
  task = task_multiblock_synthetic(task_type = "clust", n = 18L, seed = 11L)
  po_mb = PipeOpMBsPLS$new(blocks = task$block_features(), param_vals = list(ncomp = 1L, append = FALSE))
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )

  inst = mlr3tuning::ti(
    task = task,
    learner = gl,
    resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("clust.dunn"),
    terminator = bbotk::trm("none")
  )

  expect_error(
    TunerSeqMBsPLS$new(budget = 1L, resampling = mlr3::rsmp("holdout"), early_stopping = FALSE)$optimize(inst),
    "supports only MB-sPLS measures"
  )
})


test_that("TunerSeqMBsPCA rejects unsupported measures", {
  task = task_multiblock_synthetic(task_type = "clust", n = 18L, seed = 12L)
  po_mb = PipeOpMBsPCA$new(blocks = task$block_features(), param_vals = list(ncomp = 1L))
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )

  inst = mlr3tuning::ti(
    task = task,
    learner = gl,
    resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("clust.dunn"),
    terminator = bbotk::trm("none")
  )

  expect_error(
    TunerSeqMBsPCA$new(budget = 1L, resampling = mlr3::rsmp("holdout"), early_stopping = FALSE)$optimize(inst),
    "requires the measure 'mbspca.mean_ev'|supports only the measure 'mbspca.mean_ev'"
  )
})


test_that("TunerSeqMBsPLS can optimize each package MB-sPLS measure on a tiny task", {
  task = task_multiblock_synthetic(task_type = "clust", n = 20L, seed = 21L)
  blocks = task$block_features()
  mids = c("mbspls.mac_evwt", "mbspls.mac", "mbspls.ev", "mbspls.block_ev")

  for (mid in mids) {
    po_mb = PipeOpMBsPLS$new(
      blocks = blocks,
      param_vals = list(ncomp = 1L, log_env = new.env(parent = emptyenv()), append = FALSE)
    )
    gl = mlr3::as_learner(
      po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
    )

    inst = mlr3tuning::ti(
      task = task,
      learner = gl,
      resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr(mid),
      terminator = bbotk::trm("none")
    )

    expect_no_error(
      TunerSeqMBsPLS$new(budget = 1L, resampling = mlr3::rsmp("holdout"), early_stopping = FALSE)$optimize(inst)
    )
    expect_true(is.matrix(inst$result_learner_param_vals$c_matrix))
    expect_named(inst$result_y, mid)
  }
})


test_that("TunerSeqMBsPCA can optimize mbspca.mean_ev on a tiny task", {
  task = task_multiblock_synthetic(task_type = "clust", n = 20L, seed = 22L)
  po_mb = PipeOpMBsPCA$new(
    blocks = task$block_features(),
    param_vals = list(ncomp = 1L, log_env = new.env(parent = emptyenv()))
  )
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )

  inst = mlr3tuning::ti(
    task = task,
    learner = gl,
    resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("mbspca.mean_ev"),
    terminator = bbotk::trm("none")
  )

  expect_no_error(
    TunerSeqMBsPCA$new(budget = 1L, resampling = mlr3::rsmp("holdout"), early_stopping = FALSE)$optimize(inst)
  )
  expect_true(is.matrix(inst$result_learner_param_vals$c_matrix))
  expect_named(inst$result_y, "mbspca.mean_ev")
})
