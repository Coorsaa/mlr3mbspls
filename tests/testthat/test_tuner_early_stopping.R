early_stop_learner = function(method, blocks, ncomp) {
  component = if (method == "pls") {
    PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = ncomp, append = FALSE))
  } else {
    PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = ncomp))
  }
  mlr3::as_learner(mlr3pipelines::po("scale") %>>% component %>>%
    mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
}

early_stop_run = function(task, method, ncomp, rs, seed = 3L, ...) {
  blocks = task$block_features()
  instance = mlr3tuning::ti(
    task = task, learner = early_stop_learner(method, blocks, ncomp),
    resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr(if (method == "pls") "mbspls.mac_evwt" else "mbspca.mean_ev"),
    terminator = bbotk::trm("evals", n_evals = 1L)
  )
  constructor = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
  tuner = constructor$new(budget = 2L, resampling = rs, ...)
  set.seed(seed)
  tuner$optimize(instance)
  list(instance = instance, tuner = tuner)
}

early_stop_task = function(n = 45L, seed = 29L) {
  task = task_multiblock_synthetic(task_type = "clust", n = n, seed = seed)
  task$select(unique(unlist(task$block_features())))
  task
}

test_that("TunerSeqMBsPLS early stopping drops the failing component and its payloads", {
  task = early_stop_task()
  rs = mlr3::rsmp("cv", folds = 3L)
  rs$instantiate(task)
  # Deterministic fold p-values: component 1 passes, component 2 fails.
  calls = new.env(parent = emptyenv())
  calls$n = 0L
  testthat::local_mocked_bindings(
    cpp_perm_test_oos = function(...) {
      calls$n = calls$n + 1L
      list(p_value = if (calls$n <= 3L) 0.001 else 0.9)
    },
    .package = "mlr3mbspls"
  )
  stopped = early_stop_run(task, "pls", ncomp = 3L, rs, early_stopping = TRUE, n_perm = 19L)
  reference = early_stop_run(task, "pls", ncomp = 1L, rs, early_stopping = FALSE)

  expect_identical(calls$n, 6L)
  c_matrix = stopped$instance$result_learner_param_vals$c_matrix
  expect_identical(colnames(c_matrix), "LC1")
  expect_equal(c_matrix, reference$instance$result_learner_param_vals$c_matrix)
  expect_equal(stopped$instance$result_y, reference$instance$result_y)

  components = stopped$tuner$diagnostics$components
  expect_identical(components$kept, c(TRUE, FALSE))
  expect_true(components$p_combined[1L] < 0.05 && components$p_combined[2L] > 0.05)
  expect_identical(components$p_sequential, cummax(components$p_combined))
  stats = stopped$tuner$diagnostics$fold_statistics
  expect_identical(stats$p_value, rep(c(0.001, 0.9), each = 3L))
  expect_identical(stats$kept, rep(c(TRUE, FALSE), each = 3L))
})

test_that("TunerSeqMBsPLS early stopping keeps LC1 at a zero cutoff and all components at one", {
  task = early_stop_task()
  rs = mlr3::rsmp("cv", folds = 3L)
  rs$instantiate(task)
  none = early_stop_run(task, "pls", ncomp = 2L, rs, early_stopping = TRUE,
    n_perm = 9L, perm_alpha = 0)
  expect_identical(colnames(none$instance$result_learner_param_vals$c_matrix), "LC1")
  expect_true(all(is.finite(none$tuner$diagnostics$fold_statistics$p_value)))

  all_kept = early_stop_run(task, "pls", ncomp = 2L, rs, early_stopping = TRUE,
    n_perm = 9L, perm_alpha = 1)
  expect_identical(colnames(all_kept$instance$result_learner_param_vals$c_matrix), c("LC1", "LC2"))
  expect_identical(all_kept$tuner$diagnostics$components$kept, c(TRUE, TRUE))
})

test_that("TunerSeqMBsPCA early stopping stops on the cross-block diagnostic", {
  task = early_stop_task()
  rs = mlr3::rsmp("cv", folds = 3L)
  rs$instantiate(task)
  stopped = early_stop_run(task, "pca", ncomp = 2L, rs, early_stopping = TRUE,
    n_perm = 9L, perm_alpha = 0)
  expect_identical(colnames(stopped$instance$result_learner_param_vals$c_matrix), "PC1")
  expect_true(is.finite(stopped$tuner$diagnostics$components$p_value[1L]))
})

test_that("TunerSeqMBsPCA skips early stopping for a single block and keeps all components", {
  set.seed(41L)
  n = 60L
  scores = matrix(stats::rnorm(n * 3L), n, 3L)
  loadings = rbind(
    c(3, 0, 0), c(3, 0, 0), c(3, 0, 0),
    c(0, 2.5, 0), c(0, 2.5, 0), c(0, 2.5, 0),
    c(0, 0, 2), c(0, 0, 2), c(0, 0, 2)
  )
  x = scores %*% t(loadings) + matrix(stats::rnorm(n * 9L, sd = 0.5), n, 9L)
  colnames(x) = paste0("x", seq_len(9L))
  task = mlr3cluster::TaskClust$new("one_block", data.table::as.data.table(x))
  blocks = list(only = colnames(x))
  gl = mlr3::as_learner(PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 3L)) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
  instance = mlr3tuning::ti(
    task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("mbspca.mean_ev"), terminator = bbotk::trm("evals", n_evals = 1L)
  )
  expect_warning(
    TunerSeqMBsPCA$new(budget = 1L, resampling = mlr3::rsmp("cv", folds = 2L),
      early_stopping = TRUE, n_perm = 9L)$optimize(instance),
    "at least two blocks"
  )
  expect_identical(ncol(instance$result_learner_param_vals$c_matrix), 3L)
  expect_true(is.finite(instance$result_y))
})
