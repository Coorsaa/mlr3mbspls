test_that("sequential tuning EV uses the original held-out variance denominator", {
  z = as.numeric(scale(seq_len(20L)))
  u = as.numeric(residuals(lm(rep(c(-1, 1), 10L) ~ z)))
  u = u * sqrt(sum(z^2) / sum(u^2))
  X = list(a = cbind(strong = 10 * z, weak = u), b = cbind(strong = 10 * z, weak = u))
  W = list(list(c(1, 0), c(1, 0)), list(c(0, 1), c(0, 1)))
  original_ss = vapply(X, function(block) sum(block^2), numeric(1L))

  for (constructor in list(TunerSeqMBsPLS, TunerSeqMBsPCA)) {
    tuner = constructor$new(early_stopping = FALSE)
    private = tuner$.__enclos_env__$private
    residual = if (inherits(tuner, "TunerSeqMBsPLS")) {
      private$.deflate_blocks_split(X, NULL, W[[1L]])$train
    } else {
      private$.deflate_blocks(X, W[[1L]])
    }
    args = list(X_train_fit = residual, X_test = residual,
      W_list = W[[2L]], original_ss = original_ss)
    if (inherits(tuner, "TunerSeqMBsPLS")) args$correlation_method = "pearson"
    payload = do.call(private$.one_lv_payload, args)

    P = list(private$.compute_train_loadings(X, W[[1L]]),
      private$.compute_train_loadings(residual, W[[2L]]))
    full = compute_test_ev(X, W, P, loading_source = "train")
    expect_equal(as.numeric(payload$ev_comp), 1 / 101, tolerance = 1e-10)
    expect_equal(as.numeric(payload$ev_block), as.numeric(full$ev_block[2L, ]))
    expect_equal(as.numeric(payload$ev_comp), as.numeric(full$ev_comp[2L]))
  }
})

test_that("sequential tuners derive feature schemas inside each training fold", {
  set.seed(701L)
  data = as.data.frame(matrix(rnorm(24L * 6L), 24L,
    dimnames = list(NULL, c("x1", "x2", "x3", "z1", "z2", "z3"))))
  task = mlr3cluster::TaskClust$new("fold_schema", data)
  blocks = list(x = c("x1", "x2", "x3"), z = c("z1", "z2", "z3"))
  # Simulate a learned filter whose selected columns differ across fit samples.
  selector = function(task) {
    if (task$nrow > 12L) c("x1", "x2", "z1", "z2") else c("x2", "z2")
  }
  for (method in c("pls", "pca")) {
    component = if (method == "pls") {
      PipeOpMBsPLS$new(blocks = blocks)
    } else {
      PipeOpMBsPCA$new(blocks = blocks, param_vals = list(max_iter = 7L, tol = 2e-7))
    }
    learner = mlr3::as_learner(
      mlr3pipelines::po("select", selector = selector) %>>% component %>>%
        mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
    )
    instance = mlr3tuning::ti(
      task = task, learner = learner, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr(if (method == "pls") "mbspls.mac" else "mbspca.mean_ev"),
      terminator = bbotk::trm("evals", n_evals = 1L)
    )
    constructor = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
    expect_no_error(constructor$new(budget = 1L,
      resampling = mlr3::rsmp("cv", folds = 2L),
      early_stopping = FALSE)$optimize(instance))
    # Fold training has one retained column per block, despite wider full data.
    expect_equal(as.numeric(instance$result_learner_param_vals$c_matrix), c(1, 1))
    expect_true(is.finite(instance$result_y))
  }
})

tuner_fold_learner = function(method, blocks, ncomp, pre = NULL) {
  log_env = new.env(parent = emptyenv())
  log_env$warn_overwrite = FALSE
  component = if (method == "pls") {
    PipeOpMBsPLS$new(blocks = blocks,
      param_vals = list(ncomp = ncomp, append = FALSE, log_env = log_env))
  } else {
    PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = ncomp, log_env = log_env))
  }
  graph = component %>>% mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
  if (!is.null(pre)) {
    graph = pre %>>% graph
  }
  mlr3::as_learner(graph)
}

tuner_fold_run = function(task, learner, method, measure, rs, budget = 3L, seed = 2L, ...) {
  instance = mlr3tuning::ti(
    task = task, learner = learner, resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr(measure), terminator = bbotk::trm("evals", n_evals = 1L)
  )
  constructor = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
  tuner = constructor$new(budget = budget, resampling = rs, ...)
  set.seed(seed)
  tuner$optimize(instance)
  list(instance = instance, tuner = tuner)
}

test_that("sequential tuners centre every inner fold by its training means", {
  task = task_multiblock_synthetic(task_type = "clust", n = 45L, seed = 5L)
  blocks = task$block_features()
  data = task$data(cols = unlist(blocks))
  shifted = data.table::copy(data)
  for (j in seq_along(shifted)) {
    data.table::set(shifted, j = j, value = shifted[[j]] + 50 + j)
  }
  rs = mlr3::rsmp("cv", folds = 3L)
  rs$instantiate(mlr3cluster::TaskClust$new("raw", data))
  cases = list(c("pls", "mbspls.mac_evwt"), c("pls", "mbspls.ev"), c("pca", "mbspca.mean_ev"))
  for (case in cases) {
    runs = lapply(list(data, shifted), function(d) {
      tuner_fold_run(mlr3cluster::TaskClust$new("fold_centring", d),
        tuner_fold_learner(case[[1L]], blocks, ncomp = 2L), case[[1L]], case[[2L]], rs,
        early_stopping = FALSE)
    })
    # Explained variance and deflation depend on centring; column offsets must
    # therefore leave the tuned matrix and the inner score unchanged.
    expect_equal(runs[[2L]]$instance$result_y, runs[[1L]]$instance$result_y, tolerance = 1e-8)
    expect_equal(runs[[2L]]$instance$result_learner_param_vals$c_matrix,
      runs[[1L]]$instance$result_learner_param_vals$c_matrix)
  }
})

test_that("TunerSeqMBsPLS reports the score of the fits that selected the candidate", {
  task = task_multiblock_synthetic(task_type = "clust", n = 45L, seed = 7L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  rs = mlr3::rsmp("cv", folds = 3L)
  rs$instantiate(task)
  for (ncomp in 1:2) {
    learner = tuner_fold_learner("pls", blocks, ncomp = ncomp,
      pre = mlr3pipelines::as_graph(mlr3pipelines::po("scale")))
    run = tuner_fold_run(task, learner, "pls", "mbspls.mac", rs, budget = 4L,
      early_stopping = FALSE)
    y = unname(run$instance$result_y)
    diagnostics = run$tuner$diagnostics
    if (ncomp == 1L) {
      # The deterministic solver reproduces the search fit of the selected c.
      expect_equal(y, diagnostics$components$search_score, tolerance = 1e-12)
    }
    stats = diagnostics$fold_statistics
    expect_equal(nrow(stats), 3L * ncomp)
    expect_true(all(stats$converged))
    expect_true(is.integer(stats$iterations) && all(stats$iterations >= 1L))

    # The same folds scored through the tuned learner give the reported score.
    tuned = learner$clone(deep = TRUE)
    tuned$param_set$values$mbspls.c_matrix = run$instance$result_learner_param_vals$c_matrix
    rr = mlr3::resample(task, tuned, rs, store_models = TRUE)
    expect_equal(unname(rr$aggregate(mlr3::msr("mbspls.mac"))), y, tolerance = 1e-6)
  }
})

test_that("TunerSeqMBsPLS fits components only on inner training folds and never refits", {
  task = task_multiblock_synthetic(task_type = "clust", n = 45L, seed = 9L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  rs = mlr3::rsmp("cv", folds = 3L)
  rs$instantiate(task)
  fit_one_lv = get("cpp_mbspls_one_lv", envir = asNamespace("mlr3mbspls"))
  calls = new.env(parent = emptyenv())
  calls$n_rows = integer()
  testthat::local_mocked_bindings(
    cpp_mbspls_one_lv = function(X_blocks, ...) {
      calls$n_rows = c(calls$n_rows, nrow(X_blocks[[1L]]))
      fit_one_lv(X_blocks, ...)
    },
    .package = "mlr3mbspls"
  )
  run = tuner_fold_run(task, tuner_fold_learner("pls", blocks, ncomp = 2L), "pls",
    "mbspls.mac", rs, budget = 1L, early_stopping = FALSE)
  expect_equal(ncol(run$instance$result_learner_param_vals$c_matrix), 2L)
  # One candidate per component, fitted once per inner fold: no refit of the
  # selected candidate and no discarded full-data fit.
  expect_identical(calls$n_rows, rep(30L, 6L))
})

test_that("TunerSeqMBsPLS warns once about retained components that did not converge", {
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 3L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  fit_one_lv = get("cpp_mbspls_one_lv", envir = asNamespace("mlr3mbspls"))
  testthat::local_mocked_bindings(
    cpp_mbspls_one_lv = function(...) {
      fit = fit_one_lv(...)
      fit$converged = FALSE
      fit$iterations = 600L
      fit
    },
    .package = "mlr3mbspls"
  )
  run = NULL
  expect_warning(
    {
      run = tuner_fold_run(task, tuner_fold_learner("pls", blocks, ncomp = 1L), "pls",
        "mbspls.mac", mlr3::rsmp("cv", folds = 2L), budget = 1L, early_stopping = FALSE)
    },
    "did not converge.*LC1 \\(inner folds 1, 2\\)"
  )
  stats = run$tuner$diagnostics$fold_statistics
  expect_false(any(stats$converged))
  expect_identical(stats$iterations, c(600L, 600L))
})

test_that("parallel inner folds propose the same candidates as a sequential run", {
  testthat::skip_if_not_installed("future")
  testthat::skip_if_not_installed("future.apply")
  # One available core: future evaluates in the main process, so the test
  # spawns no workers but still goes through future_lapply().
  withr::local_options(mc.cores = 1L)
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 7L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  rs = mlr3::rsmp("cv", folds = 2L)
  rs$instantiate(task)
  run = function(method, parallel) {
    component = if (method == "pls") {
      PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L, append = FALSE))
    } else {
      PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L))
    }
    gl = mlr3::as_learner(component %>>%
      mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
    instance = mlr3tuning::ti(
      task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr(if (method == "pls") "mbspls.mac" else "mbspca.mean_ev"),
      terminator = bbotk::trm("evals", n_evals = 1L)
    )
    constructor = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
    withr::local_seed(5L)
    constructor$new(budget = 4L, resampling = rs, early_stopping = FALSE, parallel = parallel)$optimize(instance)
    list(
      c_matrix = instance$result_learner_param_vals$c_matrix,
      y = instance$result_y,
      seed = get(".Random.seed", envir = globalenv())
    )
  }
  for (method in c("pls", "pca")) {
    expect_identical(run(method, "inner"), run(method, "none"))
  }
})

test_that("TunerSeqMBsPCA warns once about retained components that did not converge", {
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 4L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  fit_one_lv = cpp_mbspca_one_lv
  testthat::local_mocked_bindings(
    cpp_mbspca_one_lv = function(...) {
      fit = fit_one_lv(...)
      fit$converged = FALSE
      fit
    },
    .package = "mlr3mbspls"
  )
  run = NULL
  expect_warning(
    {
      run = tuner_fold_run(task, tuner_fold_learner("pca", blocks, ncomp = 1L), "pca",
        "mbspca.mean_ev", mlr3::rsmp("cv", folds = 2L), budget = 1L, early_stopping = FALSE)
    },
    "MB-sPCA solver did not converge.*PC1 \\(inner folds 1, 2\\)"
  )
  expect_false(any(run$tuner$diagnostics$fold_statistics$converged))
})
