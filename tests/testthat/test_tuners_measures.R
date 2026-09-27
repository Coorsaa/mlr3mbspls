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


test_that("TunerSeqMBsPLS errors if its performance_metric disagrees with the learner", {
  task = task_multiblock_synthetic(task_type = "clust", n = 18L, seed = 23L)
  po_mb = PipeOpMBsPLS$new(
    blocks = task$block_features(),
    param_vals = list(ncomp = 1L, performance_metric = "frobenius", append = FALSE)
  )
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )

  inst = mlr3tuning::ti(
    task = task,
    learner = gl,
    resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("mbspls.mac"),
    terminator = bbotk::trm("none")
  )

  expect_error(
    TunerSeqMBsPLS$new(
      budget = 1L,
      resampling = mlr3::rsmp("holdout"),
      early_stopping = FALSE,
      performance_metric = "mac"
    )$optimize(inst),
    "performance_metric .* must match"
  )
})


test_that("TunerSeqMBsPLS reads performance_metric from its parameter set", {
  task = task_multiblock_synthetic(task_type = "clust", n = 18L, seed = 24L)
  tuned_instance = function(learner_metric) {
    po_mb = PipeOpMBsPLS$new(
      blocks = task$block_features(),
      param_vals = list(ncomp = 1L, performance_metric = learner_metric, append = FALSE)
    )
    gl = mlr3::as_learner(
      po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
    )
    mlr3tuning::ti(
      task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr("mbspls.mac"), terminator = bbotk::trm("none")
    )
  }

  tuner = TunerSeqMBsPLS$new(budget = 1L, resampling = mlr3::rsmp("holdout"), early_stopping = FALSE)
  expect_identical(tuner$param_set$values$performance_metric, "mac")
  tuner$param_set$values$performance_metric = "frobenius"
  expect_no_error(tuner$optimize(tuned_instance("frobenius")))
  expect_error(
    tuner$optimize(tuned_instance("mac")),
    "performance_metric \\('frobenius'\\) must match the PipeOpMBsPLS node \\('mac'\\)"
  )
  expect_identical(
    TunerSeqMBsPLS$new(performance_metric = "frobenius")$param_set$values$performance_metric,
    "frobenius"
  )
})


test_that("TunerSeqMBsPLS explains when every candidate score is undefined", {
  task = task_multiblock_synthetic(task_type = "clust", n = 18L, seed = 25L)$filter(1:4)
  po_mb = PipeOpMBsPLS$new(blocks = task$block_features(), param_vals = list(ncomp = 1L, append = FALSE))
  gl = mlr3::as_learner(
    po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )
  inst = mlr3tuning::ti(
    task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("mbspls.mac_evwt"), terminator = bbotk::trm("none")
  )
  # A single validation row leaves every latent correlation undefined.
  expect_error(
    suppressWarnings(
      TunerSeqMBsPLS$new(budget = 1L, resampling = mlr3::rsmp("holdout"), early_stopping = FALSE)$optimize(inst)
    ),
    "All 1 candidate c-vectors produced an undefined 'mbspls.mac_evwt' score for component 1"
  )
})


tuner_error_instance = function(method, seed) {
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = seed)
  component = if (method == "pls") {
    PipeOpMBsPLS$new(blocks = task$block_features(), param_vals = list(ncomp = 1L, append = FALSE))
  } else {
    PipeOpMBsPCA$new(blocks = task$block_features(), param_vals = list(ncomp = 1L))
  }
  gl = mlr3::as_learner(
    component %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  )
  mlr3tuning::ti(
    task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr(if (method == "pls") "mbspls.mac" else "mbspca.mean_ev"),
    terminator = bbotk::trm("none")
  )
}


test_that("sequential tuners report errors raised while scoring a candidate", {
  testthat::local_mocked_bindings(
    cpp_mbspls_one_lv = function(...) stop("solver failure sentinel"),
    cpp_mbspca_one_lv = function(...) stop("pca solver failure sentinel"),
    .package = "mlr3mbspls"
  )
  expect_error(
    TunerSeqMBsPLS$new(budget = 2L, resampling = mlr3::rsmp("cv", folds = 2L), early_stopping = FALSE)$optimize(
      tuner_error_instance("pls", 26L)
    ),
    "TunerSeqMBsPLS failed while scoring component 1: solver failure sentinel"
  )
  expect_error(
    TunerSeqMBsPCA$new(budget = 2L, resampling = mlr3::rsmp("cv", folds = 2L), early_stopping = FALSE)$optimize(
      tuner_error_instance("pca", 26L)
    ),
    "TunerSeqMBsPCA failed while scoring component 1: pca solver failure sentinel"
  )
})


test_that("TunerSeqMBsPLS reports a later candidate's error after undefined scores", {
  fit_one_lv = cpp_mbspls_one_lv
  calls = new.env(parent = emptyenv())
  calls$fit = 0L
  testthat::local_mocked_bindings(
    # The first candidate is undefined, the second one fails in the solver.
    mbspls_measure_score_diagnostics = function(...) {
      list(score = NA_real_, defined = FALSE, message = "undefined", n_undefined_components = 1L)
    },
    cpp_mbspls_one_lv = function(...) {
      calls$fit = calls$fit + 1L
      if (calls$fit > 1L) stop("second candidate sentinel")
      fit_one_lv(...)
    },
    .package = "mlr3mbspls"
  )
  expect_error(
    TunerSeqMBsPLS$new(tuner = "grid_search", budget = 2L, resampling = mlr3::rsmp("holdout"),
      early_stopping = FALSE)$optimize(tuner_error_instance("pls", 27L)),
    "second candidate sentinel"
  )
  expect_identical(calls$fit, 2L)
})


test_that("TunerSeqMBsPCA ranks undefined candidates last and reuses the selected fits", {
  score = mbspca_measure_score_from_payload
  fit_one_lv = cpp_mbspca_one_lv
  calls = new.env(parent = emptyenv())
  calls$score = 0L
  calls$fit = 0L
  testthat::local_mocked_bindings(
    # Both inner folds of the first candidate have an undefined score.
    mbspca_measure_score_from_payload = function(...) {
      calls$score = calls$score + 1L
      if (calls$score <= 2L) NA_real_ else score(...)
    },
    cpp_mbspca_one_lv = function(...) {
      calls$fit = calls$fit + 1L
      fit_one_lv(...)
    },
    .package = "mlr3mbspls"
  )
  inst = tuner_error_instance("pca", 28L)
  tuner = TunerSeqMBsPCA$new(tuner = "grid_search", budget = 3L,
    resampling = mlr3::rsmp("cv", folds = 2L), early_stopping = FALSE)
  expect_no_error(tuner$optimize(inst))
  expect_identical(tuner$diagnostics$components$n_evals, 3L)
  expect_true(is.finite(inst$result_y))
  expect_equal(unname(inst$result_y), tuner$diagnostics$components$search_score)
  # Three candidates on two folds, and no refit of the selected candidate.
  expect_identical(calls$fit, 6L)
})


test_that("TunerSeqMBsPCA explains when every candidate score is undefined", {
  testthat::local_mocked_bindings(
    mbspca_measure_score_from_payload = function(...) NA_real_,
    .package = "mlr3mbspls"
  )
  expect_error(
    TunerSeqMBsPCA$new(tuner = "grid_search", budget = 2L, resampling = mlr3::rsmp("cv", folds = 2L),
      early_stopping = FALSE)$optimize(tuner_error_instance("pca", 29L)),
    "All 2 candidate c-vectors produced an undefined 'mbspca.mean_ev' score for component 1"
  )
})
