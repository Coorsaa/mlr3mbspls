test_that("PipeOpMBsPLSBootstrapSelect - stability_only TRUE is a pass-through", {
  set.seed(1)

  n = 40
  p1 = 6
  p2 = 8

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())

  po_mbspls = mlr3mbspls::PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      store_train_blocks = TRUE,
      append = FALSE
    )
  )

  task_lv = po_mbspls$train(list(task))[[1]]
  dt_before = as.data.frame(task_lv$data())
  feats_before = task_lv$feature_names

  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = TRUE,
      B = 5L,
      selection_method = "ci",
      align = "block_sign",
      workers = 1L
    )
  )

  out_train = po_sel$train(list(task_lv))[[1]]

  # Pass-through means: no feature dropping and no LV replacement.
  expect_setequal(out_train$feature_names, feats_before)
  expect_equal(as.data.frame(out_train$data()), dt_before)

  # Predict should also pass-through unchanged in stability_only mode.
  task_lv_new = task_lv$clone(deep = TRUE)
  task_lv_new$filter(1:10)
  out_pred = po_sel$predict(list(task_lv_new))[[1]]
  expect_setequal(out_pred$feature_names, task_lv_new$feature_names)
  expect_equal(as.data.frame(out_pred$data()), as.data.frame(task_lv_new$data()))
})


test_that("PipeOpMBsPLSBootstrapSelect - errors if blocks cannot be rebuilt and X_train_blocks is missing", {
  set.seed(2)

  n = 30
  p1 = 5
  p2 = 7

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())

  # Train MB-sPLS *without* storing raw training blocks.
  po_mbspls = mlr3mbspls::PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      store_train_blocks = FALSE,
      append = FALSE
    )
  )

  task_lv = po_mbspls$train(list(task))[[1]]
  # At this stage, block features are no longer present in the task backend.

  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = FALSE,
      B = 3L,
      workers = 1L
    )
  )

  expect_error(
    po_sel$train(list(task_lv)),
    "Cannot rebuild training blocks"
  )
})


test_that("PipeOpMBsPLSBootstrapSelect stores the upstream run_id", {
  set.seed(12)

  n = 24
  b1 = matrix(rnorm(n * 4), nrow = n, ncol = 4)
  colnames(b1) = paste0("x", seq_len(ncol(b1)))
  b2 = matrix(rnorm(n * 5), nrow = n, ncol = 5)
  colnames(b2) = paste0("z", seq_len(ncol(b2)))
  y = rnorm(n)

  task = mlr3::TaskRegr$new(id = "bootsel_runid", backend = data.frame(b1, b2, y = y), target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  log_env = new.env(parent = emptyenv())

  po_mbspls = mlr3mbspls::PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      store_train_blocks = TRUE,
      append = FALSE
    )
  )

  task_lv = po_mbspls$train(list(task))[[1L]]

  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = TRUE,
      stability_only = TRUE,
      B = 3L,
      workers = 1L
    )
  )

  po_sel$train(list(task_lv))
  expect_identical(po_sel$state$run_id, log_env$mbspls_state$run_id)
})


test_that("PipeOpMBsPLSBootstrapSelect requires bootstrap when stability_only is TRUE", {
  log_env = new.env(parent = emptyenv())
  log_env$mbspls_state = list(
    blocks = list(block = "x"),
    weights = list()
  )

  po_sel = mlr3mbspls::PipeOpMBsPLSBootstrapSelect$new(
    param_vals = list(
      log_env = log_env,
      bootstrap = FALSE,
      stability_only = TRUE
    )
  )

  task = mlr3::TaskRegr$new(
    id = "bootstrap_guard",
    backend = data.frame(LV1_block = rnorm(10), y = rnorm(10)),
    target = "y"
  )

  expect_error(
    po_sel$train(list(task)),
    "stability_only=TRUE requires bootstrap=TRUE"
  )
})

test_that("PipeOpMBsPLSBootstrapSelect uses grouped deterministic streams", {
  set.seed(31)
  n = 24L
  block1 = matrix(rnorm(n * 3), nrow = n,
    dimnames = list(NULL, paste0("x", 1:3)))
  block2 = matrix(rnorm(n * 3), nrow = n,
    dimnames = list(NULL, paste0("z", 1:3)))
  task = mlr3::TaskRegr$new(
    "grouped_bootstrap",
    data.frame(block1, block2, y = rnorm(n)),
    target = "y"
  )
  blocks = list(b1 = colnames(block1), b2 = colnames(block2))
  log_env = new.env(parent = emptyenv())
  upstream = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(
    ncomp = 1L,
    c_b1 = sqrt(3),
    c_b2 = sqrt(3),
    log_env = log_env,
    store_train_blocks = TRUE,
    seed_train = 32L
  ))
  score_task = upstream$train(list(task))[[1L]]
  groups = stats::setNames(rep(paste0("participant_", 1:12), each = 2),
    score_task$row_ids)

  selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env,
    bootstrap = TRUE,
    stability_only = TRUE,
    bootstrap_groups = groups,
    B = 4L,
    seed_bootstrap = 33L,
    workers = 1L
  ))
  before = .Random.seed
  selector$train(list(score_task))

  expect_true(selector$state$bootstrap_grouped)
  expect_identical(selector$state$n_exchangeability_units, 12L)
  expect_length(selector$state$rng_streams, 4L)
  expect_true("replicates_effective" %in% names(selector$state$weights_ci))
  expect_identical(.Random.seed, before)

  bad_selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env,
    bootstrap = TRUE,
    stability_only = TRUE,
    bootstrap_groups = groups[-1L],
    B = 2L,
    seed_bootstrap = 34L,
    workers = 1L
  ))
  expect_error(bad_selector$train(list(score_task)), "missing task row IDs")

  one_replicate = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env,
    bootstrap = TRUE,
    stability_only = TRUE,
    B = 1L,
    workers = 1L
  ))
  expect_error(
    one_replicate$train(list(score_task)),
    "at least two replicates"
  )

  boundary_alpha = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env,
    bootstrap = TRUE,
    stability_only = TRUE,
    B = 2L,
    alpha = 0,
    workers = 1L
  ))
  expect_error(
    boundary_alpha$train(list(score_task)),
    "strictly between 0 and 1"
  )
})

test_that("bootstrap component matching uses sequentially deflated scores", {
  z = as.numeric(scale(seq_len(20L)))
  u = rep(c(-1, 1), 10L)
  u = as.numeric(residuals(lm(u ~ z)))
  x = cbind(z = z, mixed = 10 * z + u)
  X = list(a = x, b = x)
  W_ref = list(
    list(a = c(z = 1, mixed = 0), b = c(z = 1, mixed = 0)),
    list(a = c(z = 0, mixed = 1), b = c(z = 0, mixed = 1))
  )

  # A valid two-component deflation path with reversed loading directions.
  # The first score still matches reference component 1; the second is
  # negatively aligned to reference component 2 after deflation.
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = function(X_blocks, ...) {
      t1 = X_blocks[[1L]][, 2L]
      t2 = X_blocks[[1L]][, 1L] -
        t1 * sum(t1 * X_blocks[[1L]][, 1L]) / sum(t1^2)
      list(W = rev(W_ref), T_mat = cbind(t1, t1, t2, t2))
    },
    .package = "mlr3mbspls"
  )
  selector = PipeOpMBsPLSBootstrapSelect$new()
  result = selector$.__enclos_env__$private$.bootstrap_align_and_summarise(
    X_list = X,
    W_ref = W_ref,
    blocks = list(a = colnames(x), b = colnames(x)),
    ncomp = 2L,
    sparsity = list(type = "c_vec", c_vec = c(a = sqrt(2), b = sqrt(2))),
    B = 2L,
    align = "score_correlation",
    rng_streams = mb_rng_streams(2L, 66L),
    min_score_cor = 0.8
  )
  summary = result$summary
  first = summary[summary$component == "LC_01" & summary$feature == "mixed", ]
  second = summary[summary$component == "LC_02" & summary$feature == "z", ]
  expect_equal(first$boot_mean, c(1, 1))
  expect_equal(second$boot_mean, c(-1, -1))
  expect_equal(result$n_eff_by_component$n_eff, c(2L, 2L))
})

test_that("seeded parallel bootstrap matches sequential results and restores RNG", {
  skip_on_cran()
  skip_if_not_installed("future")
  skip_if_not_installed("future.apply")
  skip_if_not(future::supportsMulticore())
  old_plan = future::plan()
  withr::defer(future::plan(old_plan))
  future::plan(future::multicore, workers = 2L)

  set.seed(67L)
  X = list(
    a = matrix(rnorm(80), 20L, dimnames = list(NULL, paste0("a", 1:4))),
    b = matrix(rnorm(80), 20L, dimnames = list(NULL, paste0("b", 1:4)))
  )
  fit = mlr3mbspls:::cpp_mbspls_multi_lv(X, c(2, 2), K = 2L)
  W = lapply(fit$W, function(wk) {
    stats::setNames(lapply(seq_along(X), function(b) {
      stats::setNames(as.numeric(wk[[b]]), colnames(X[[b]]))
    }), names(X))
  })
  selector = PipeOpMBsPLSBootstrapSelect$new()
  run = function(workers) {
    selector$.__enclos_env__$private$.bootstrap_align_and_summarise(
      X_list = X, W_ref = W, blocks = lapply(X, colnames), ncomp = 2L,
      sparsity = list(type = "c_vec", c_vec = c(a = 2, b = 2)),
      B = 3L, workers = workers, rng_streams = mb_rng_streams(3L, 68L)
    )
  }
  before = .Random.seed
  sequential = run(1L)
  parallel_result = run(2L)
  expect_identical(.Random.seed, before)
  expect_identical(parallel_result, sequential)
})
