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

# Messages of all warnings raised while evaluating `expr`.
collect_warnings = function(expr) {
  rec = new.env()
  rec$messages = character(0)
  withCallingHandlers(expr, warning = function(w) {
    rec$messages = c(rec$messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  rec$messages
}

# Two-block toy data with named reference weights for mocked-solver tests.
bootstrap_toy_blocks = function(seed = 71L, n = 30L) {
  set.seed(seed)
  latent = rnorm(n)
  a = cbind(a1 = latent + rnorm(n, sd = 0.3), a2 = rnorm(n), a3 = rnorm(n))
  b = cbind(b1 = latent + rnorm(n, sd = 0.3), b2 = rnorm(n))
  list(
    X = list(a = scale(a, scale = FALSE), b = scale(b, scale = FALSE)),
    W_ref = list(LC_01 = list(a = c(a1 = 1, a2 = 0, a3 = 0), b = c(b1 = 1, b2 = 0)))
  )
}

run_toy_bootstrap = function(toy, align, B = 4L, ...) {
  selector = PipeOpMBsPLSBootstrapSelect$new()
  selector$.__enclos_env__$private$.bootstrap_align_and_summarise(
    X_list = toy$X,
    W_ref = toy$W_ref,
    blocks = lapply(toy$X, colnames),
    ncomp = 1L,
    sparsity = list(type = "c_vec", c_vec = c(a = 1, b = 1)),
    B = B,
    align = align,
    rng_streams = mb_rng_streams(B, 72L),
    min_score_cor = 0,
    ...
  )
}

test_that("score_correlation aligns every block separately", {
  toy = bootstrap_toy_blocks()
  # Replicate fits in which only block b is sign-flipped relative to the
  # reference: the criterion is invariant to such a flip.
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = function(X_blocks, ...) {
      w_a = c(1, 0, 0)
      w_b = c(-1, 0)
      list(
        W = list(list(w_a, w_b)),
        T_mat = cbind(X_blocks$a %*% w_a, X_blocks$b %*% w_b),
        converged = TRUE
      )
    },
    .package = "mlr3mbspls"
  )

  by_score = run_toy_bootstrap(toy, "score_correlation")
  by_weight = run_toy_bootstrap(toy, "block_sign")

  mean_of = function(res, feature) res$summary$boot_mean[res$summary$feature == feature]
  expect_equal(mean_of(by_score, "a1"), 1)
  expect_equal(mean_of(by_score, "b1"), 1)
  expect_equal(by_score$summary$ci_lower[by_score$summary$feature == "b1"], 1)
  expect_equal(by_score$summary, by_weight$summary)
})

test_that("block_sign falls back to the score correlation for disjoint supports", {
  toy = bootstrap_toy_blocks()
  toy$X$a[, "a2"] = toy$X$a[, "a1"] + rnorm(nrow(toy$X$a), sd = 0.05)
  # The replicate selects the near-duplicate a2 with a negative sign, so the
  # weight inner product with the reference (a1 only) is exactly zero.
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = function(X_blocks, ...) {
      w_a = c(0, -1, 0)
      w_b = c(1, 0)
      list(
        W = list(list(w_a, w_b)),
        T_mat = cbind(X_blocks$a %*% w_a, X_blocks$b %*% w_b),
        converged = TRUE
      )
    },
    .package = "mlr3mbspls"
  )

  res = run_toy_bootstrap(toy, "block_sign")
  a2 = res$summary[res$summary$feature == "a2", ]
  expect_equal(a2$boot_mean, 1)
  expect_gt(a2$ci_lower, 0)

  diag = res$alignment_diagnostics
  expect_identical(diag$n_fallback[diag$block == "a"], 4L)
  expect_identical(diag$n_fallback[diag$block == "b"], 0L)
  expect_identical(sum(diag$n_unresolved), 0L)
  expect_identical(diag$n_accepted, c(4L, 4L))
})

test_that("replicate refits are centred on their own resampled rows", {
  toy = bootstrap_toy_blocks()
  toy$X = lapply(toy$X, function(m) m + 5)
  rec = new.env()
  rec$means = list()
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = function(X_blocks, ...) {
      rec$means[[length(rec$means) + 1L]] = unlist(lapply(X_blocks, colMeans))
      list(
        W = list(list(c(1, 0, 0), c(1, 0))),
        T_mat = cbind(X_blocks$a[, 1L], X_blocks$b[, 1L]),
        converged = c(FALSE)
      )
    },
    .package = "mlr3mbspls"
  )

  expect_warning(
    {
      res = run_toy_bootstrap(toy, "block_sign", B = 3L)
    },
    "3 of 3 replicate refits did not converge"
  )
  expect_length(rec$means, 3L)
  expect_true(all(abs(unlist(rec$means)) < 1e-10))
  expect_identical(res$n_nonconverged_replicates, 3L)
  expect_identical(res$n_eff_by_component$n_nonconverged, 3L)
})

test_that("stratified bootstrap keeps singleton strata in every replicate", {
  set.seed(73L)
  n = 30L
  site = cbind(site_A = c(rep(1, n - 1L), 0), site_B = c(rep(0, n - 1L), 1))
  a = cbind(a1 = rnorm(n), a2 = rnorm(n))
  X = list(a = a, site = site)
  W_ref = list(LC_01 = list(a = c(a1 = 1, a2 = 0), site = c(site_A = 1, site_B = 0)))
  rec = new.env()
  rec$counts = integer(0)
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = function(X_blocks, ...) {
      x = X_blocks$site[, "site_B"]
      rec$counts[length(rec$counts) + 1L] = if (diff(range(x)) < 1e-12) {
        0L
      } else {
        sum(abs(x - max(x)) < 1e-12)
      }
      list(
        W = list(list(c(1, 0), c(1, 0))),
        T_mat = cbind(X_blocks$a[, 1L], X_blocks$site[, 1L]),
        converged = TRUE
      )
    },
    .package = "mlr3mbspls"
  )
  selector = PipeOpMBsPLSBootstrapSelect$new()
  res = selector$.__enclos_env__$private$.bootstrap_align_and_summarise(
    X_list = X, W_ref = W_ref, blocks = lapply(X, colnames), ncomp = 1L,
    sparsity = list(type = "c_vec", c_vec = c(a = 1, site = 1)),
    B = 20L, stratify_block = "site", rng_streams = mb_rng_streams(20L, 74L),
    min_score_cor = 0
  )

  expect_identical(rec$counts, rep(1L, 20L))
  expect_identical(res$strata_sizes, c(site_A = 29L, site_B = 1L))
})

test_that("stratification recovers one-hot and treatment-coded factors", {
  strata_of = getFromNamespace(".mb_strata_from_dummy_block", "mlr3mbspls")
  level = factor(rep(c("x", "y", "z"), times = c(4, 3, 3)))
  one_hot = scale(stats::model.matrix(~ level - 1))
  treatment = scale(stats::model.matrix(~level)[, -1L, drop = FALSE])
  binary = scale(cbind(level_y = as.numeric(level == "y")))

  expect_identical(as.integer(strata_of(one_hot, "g")), as.integer(level))
  expect_identical(as.integer(table(strata_of(treatment, "g"))), c(4L, 3L, 3L))
  expect_identical(
    as.character(strata_of(treatment, "g")),
    ifelse(level == "x", "(reference)", paste0("level", level))
  )
  expect_identical(nlevels(strata_of(binary, "g")), 2L)
  expect_error(strata_of(cbind(v = rnorm(10)), "g"), "not a dummy-coded factor")
  expect_error(
    strata_of(cbind(p = rep(c(1, 0), 5), q = rep(c(1, 1, 0, 0, 0), 2)), "g"),
    "more than one active indicator"
  )
  # a singleton level is a valid stratum
  expect_identical(c(table(strata_of(cbind(p = c(1, rep(0, 9))), "g"))), c(p = 1L, "(reference)" = 9L))
})

test_that("stratify_by_block is validated against the upstream blocks", {
  set.seed(75L)
  n = 40L
  grp = rep(c(0, 1), each = n / 2)
  df = data.frame(
    x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n),
    z1 = rnorm(n), z2 = rnorm(n), z3 = rnorm(n),
    grp_b = grp, const = 1, y = rnorm(n)
  )
  task = mlr3::TaskRegr$new("strata_guard", df, target = "y")
  blocks = list(b1 = c("x1", "x2", "x3"), b2 = c("z1", "z2", "z3"), grp = "grp_b", flat = "const")
  log_env = new.env(parent = emptyenv())
  upstream = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(
    ncomp = 1L, c_b1 = 1.5, c_b2 = 1.5, c_grp = 1, c_flat = 1,
    log_env = log_env, store_train_blocks = TRUE
  ))
  task_lv = upstream$train(list(task))[[1L]]
  select = function(stratify) {
    PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
      log_env = log_env, stability_only = TRUE, B = 4L, seed_bootstrap = 76L,
      stratify_by_block = stratify
    ))
  }

  expect_error(select("grp_typo")$train(list(task_lv)), "not a block of the upstream MB-sPLS fit")
  expect_error(select("flat")$train(list(task_lv)), "dropped upstream")
  expect_error(select(c("grp", "b1"))$train(list(task_lv)), "single non-empty block name")
  expect_error(select("b1")$train(list(task_lv)), "not a dummy-coded factor")

  selector = select("grp")
  suppressWarnings(selector$train(list(task_lv)))
  expect_identical(selector$state$stratify_by_block, "grp")
  expect_identical(unname(selector$state$strata_sizes), c(20L, 20L))
})

test_that("the task's group role is the default exchangeability unit", {
  set.seed(77L)
  n = 24L
  df = data.frame(
    x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n),
    z1 = rnorm(n), z2 = rnorm(n), z3 = rnorm(n),
    participant = rep(sprintf("p%02d", 1:12), each = 2),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new("group_role", df, target = "y")
  task$set_col_roles("participant", roles = "group")
  blocks = list(b1 = c("x1", "x2", "x3"), b2 = c("z1", "z2", "z3"))
  log_env = new.env(parent = emptyenv())
  upstream = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(
    ncomp = 1L, c_b1 = 1.5, c_b2 = 1.5, log_env = log_env, store_train_blocks = TRUE
  ))
  task_lv = upstream$train(list(task))[[1L]]
  select = function(groups = NULL) {
    PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
      log_env = log_env, stability_only = TRUE, B = 6L, seed_bootstrap = 78L,
      bootstrap_groups = groups
    ))
  }

  by_role = select()
  expect_false(any(grepl("exchangeability units", collect_warnings(by_role$train(list(task_lv))))))
  expect_true(by_role$state$bootstrap_grouped)
  expect_identical(by_role$state$bootstrap_group_source, "task_group_role")
  expect_identical(by_role$state$n_exchangeability_units, 12L)
  expect_false(by_role$state$few_exchangeability_units)

  explicit = select(stats::setNames(rev(df$participant), rev(task_lv$row_ids)))
  suppressWarnings(explicit$train(list(task_lv)))
  expect_identical(explicit$state$bootstrap_group_source, "explicit")
  expect_identical(explicit$state$weights_ci, by_role$state$weights_ci)

  families = select(rep(sprintf("f%d", 1:6), each = 4))
  expect_match(
    collect_warnings(families$train(list(task_lv))),
    "only 6 exchangeability units \\(groups of the explicit `bootstrap_groups`\\)",
    all = FALSE
  )
  expect_identical(families$state$n_exchangeability_units, 6L)
  expect_true(families$state$few_exchangeability_units)

  expect_error(select(seq_len(n))$train(list(task_lv)), "equal to or coarser than the group role")
})

test_that("components without stable features emit no LV columns and keep their label", {
  set.seed(79L)
  n = 60L
  latent = rnorm(n)
  b1 = scale(cbind(latent + rnorm(n, 0, 0.2), matrix(rnorm(n * 4), n)))
  b2 = scale(cbind(latent + rnorm(n, 0, 0.2), matrix(rnorm(n * 4), n)))
  colnames(b1) = paste0("x", 1:5)
  colnames(b2) = paste0("z", 1:5)
  task = mlr3::TaskRegr$new("partial", data.frame(b1, b2, y = rnorm(n)), target = "y")
  log_env = new.env(parent = emptyenv())
  upstream = PipeOpMBsPLS$new(
    blocks = list(b1 = colnames(b1), b2 = colnames(b2)),
    param_vals = list(ncomp = 2L, c_b1 = 1, c_b2 = 1, log_env = log_env,
      store_train_blocks = TRUE, append = TRUE)
  )
  task_lv = upstream$train(list(task))[[1L]]
  selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env, B = 30L, seed_bootstrap = 11L
  ))
  out = selector$train(list(task_lv))[[1L]]

  expect_identical(selector$state$kept_components, 1L)
  expect_setequal(out$feature_names, c("LV1_b1", "LV1_b2"))
  expect_identical(names(selector$state$weights_stable), c("LC_01", "LC_02"))
  expect_true(all(unlist(selector$state$weights_stable$LC_02) == 0))
  st_env = log_env$mbspls_state
  expect_identical(st_env$ncomp_stable, 1L)
  expect_identical(st_env$ncomp, 2L)
  expect_false(st_env$stability_only)
  expect_identical(selector$state$component_matching, "exact_assignment")

  pred = selector$predict(list(task_lv))[[1L]]
  expect_equal(pred$data(cols = out$feature_names), out$data(cols = out$feature_names))
})

test_that("predictions centre the blocks with the upstream training means", {
  set.seed(80L)
  n = 40L
  latent = rnorm(n)
  b1 = cbind(x1 = latent + rnorm(n, 0, 0.3), x2 = latent + rnorm(n, 0, 0.3), x3 = rnorm(n)) + 5
  b2 = cbind(z1 = latent + rnorm(n, 0, 0.3), z2 = latent + rnorm(n, 0, 0.3), z3 = rnorm(n)) - 3
  task = mlr3::TaskRegr$new("centring", data.frame(b1, b2, y = rnorm(n)), target = "y")
  log_env = new.env(parent = emptyenv())
  upstream = PipeOpMBsPLS$new(
    blocks = list(b1 = colnames(b1), b2 = colnames(b2)),
    param_vals = list(ncomp = 1L, c_b1 = sqrt(3), c_b2 = sqrt(3), log_env = log_env,
      store_train_blocks = TRUE, append = TRUE)
  )
  task_lv = upstream$train(list(task))[[1L]]
  # The upstream fit stores centred training blocks and their means.
  st = log_env$mbspls_state
  expect_false(is.null(st$center))
  expect_true(max(abs(unlist(lapply(st$X_train_blocks, colMeans)))) < 1e-10)

  selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env, B = 10L, seed_bootstrap = 81L, selection_method = "frequency"
  ))
  out = selector$train(list(task_lv))[[1L]]
  pred = selector$predict(list(task_lv))[[1L]]
  lv = out$feature_names
  expect_true(length(lv) > 0L)
  expect_equal(pred$data(cols = lv), out$data(cols = lv))
  expect_equal(selector$state$center, st$center)
})

test_that("stable weights apply the strict interval and frequency rules", {
  selector = PipeOpMBsPLSBootstrapSelect$new()
  build = selector$.__enclos_env__$private$.build_stable_from
  feats = c("f1", "f2", "f3", "f4", "f5", "f6")
  sum_df = data.frame(
    component = "LC_01", block = "b", feature = feats,
    boot_mean = c(0.5, 0.4, -0.3, 0.0005, 0.6, NA),
    ci_lower = c(0.1, 0, -0.5, 0.0001, 0.2, NA),
    ci_upper = c(0.9, 0.8, -0.1, 0.001, 0.9, NA)
  )
  freq_df = data.frame(
    component = "LC_01", block = "b", feature = feats,
    freq = c(0.6, 0.59, 1, 0.2, 0.9, 0.9)
  )
  W_train = list(LC_01 = list(b = c(f1 = 0.7, f2 = 0.1, f3 = -0.2, f4 = 0.3, f5 = 0, f6 = 0.4)))
  args = list(K = 1L, bn = "b", blocks_map = list(b = feats), sum_df = sum_df,
    freq_df = freq_df, frequency_threshold = 0.6, W_train = W_train)

  ci = do.call(build, c(list(method = "ci", weight_source = "training"), args))
  # lo == 0 and |mean| <= magnitude_threshold are not kept; NA summaries are not kept
  expect_identical(ci$selection$bootstrap_selected, c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE))
  # training weights restricted to the intersection with the training support
  expect_equal(unname(ci$W$LC_01$b), c(0.7, 0, -0.2, 0, 0, 0))
  expect_equal(ci$selection$training_weight, unname(W_train$LC_01$b))
  expect_identical(ci$kept$LC_01, "b")

  ci_mean = do.call(build, c(list(method = "ci", weight_source = "bootstrap_mean"), args))
  expect_equal(unname(ci_mean$W$LC_01$b), c(0.5, 0, -0.3, 0, 0.6, 0))

  freq = do.call(build, c(list(method = "frequency", weight_source = "training"), args))
  expect_identical(freq$selection$bootstrap_selected, c(TRUE, FALSE, TRUE, FALSE, TRUE, TRUE))
  expect_equal(unname(freq$W$LC_01$b), c(0.7, 0, -0.2, 0, 0, 0.4))

  args_no_training = args
  args_no_training$W_train = list()
  expect_error(
    do.call(build, c(list(method = "ci", weight_source = "training"), args_no_training)),
    "training weights are unavailable"
  )
})

test_that("selected features with zero training weight are recorded", {
  toy = bootstrap_toy_blocks()
  log_env = new.env(parent = emptyenv())
  log_env$mbspls_state = list(
    run_id = "zero_training",
    blocks = lapply(toy$X, colnames),
    weights = toy$W_ref,
    loadings = toy$W_ref,
    ncomp = 1L,
    X_train_blocks = toy$X,
    sparsity = list(type = "c_vec", c_vec = c(a = 1.5, b = 1))
  )
  log_env$mbspls_states = list(zero_training = log_env$mbspls_state)
  task = mlr3::TaskRegr$new(
    "zero_training",
    data.frame(LV1_a = toy$X$a[, 1L], LV1_b = toy$X$b[, 1L], y = rnorm(nrow(toy$X$a))),
    target = "y"
  )
  # Every replicate also selects a2, which the training fit leaves at zero.
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = function(X_blocks, ...) {
      w_a = c(sqrt(0.5), sqrt(0.5), 0)
      w_b = c(1, 0)
      list(W = list(list(w_a, w_b)),
        T_mat = cbind(X_blocks$a %*% w_a, X_blocks$b %*% w_b), converged = TRUE)
    },
    .package = "mlr3mbspls"
  )
  selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env, stability_only = TRUE, B = 5L, seed_bootstrap = 82L,
    selection_method = "frequency"
  ))
  expect_no_warning(selector$train(list(task)))

  expect_identical(
    as.data.frame(selector$state$selected_not_in_training),
    data.frame(component = "LC_01", block = "a", feature = "a2")
  )
  sel = selector$state$selection
  expect_true(sel$bootstrap_selected[sel$feature == "a2"])
  expect_identical(sel$stable_weight[sel$feature == "a2"], 0)
  expect_identical(selector$state$stable_weight_source, "training")
  expect_identical(selector$state$magnitude_threshold, 1e-3)
  expect_identical(selector$state$alpha, 0.05)
})

test_that("component matching is an exact deterministic assignment", {
  assign = getFromNamespace(".mb_assignment_max", "mlr3mbspls")
  # Greedy row-order matching would pick c(1, 2) with total similarity 1.0.
  expect_identical(assign(matrix(c(0.9, 0.85, 0.8, 0.1), 2L)), c(2L, 1L))
  expect_identical(assign(matrix(0.5, 3L, 3L)), assign(matrix(0.5, 3L, 3L)))
  expect_identical(assign(matrix(1)), 1L)

  permutations = function(n) {
    if (n == 1L) {
      return(matrix(1L))
    }
    p = permutations(n - 1L)
    do.call(rbind, lapply(seq_len(n), function(i) cbind(i, ifelse(p >= i, p + 1L, p))))
  }
  set.seed(83L)
  for (K in 2:6) {
    S = matrix(round(runif(K * K), 2), K)
    p = assign(S)
    expect_setequal(p, seq_len(K))
    all_perms = permutations(K)
    best = max(apply(all_perms, 1L, function(pp) sum(S[cbind(seq_len(K), pp)])))
    expect_equal(sum(S[cbind(seq_len(K), p)]), best)
  }
})

test_that("few accepted replicates are reported against min_effective_fraction", {
  toy = bootstrap_toy_blocks()
  rec = new.env()
  rec$calls = 0L
  # Only every third replicate reproduces the reference scores; the others
  # return unrelated scores and fail the acceptance gate.
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = function(X_blocks, ...) {
      rec$calls = rec$calls + 1L
      w_a = c(1, 0, 0)
      w_b = c(1, 0)
      T_mat = cbind(X_blocks$a %*% w_a, X_blocks$b %*% w_b)
      if (rec$calls %% 3L != 1L) {
        T_mat = cbind(X_blocks$a[, 3L], X_blocks$b[, 2L])
      }
      list(W = list(list(w_a, w_b)), T_mat = T_mat, converged = TRUE)
    },
    .package = "mlr3mbspls"
  )
  selector = PipeOpMBsPLSBootstrapSelect$new()
  expect_warning(
    {
      res = selector$.__enclos_env__$private$.bootstrap_align_and_summarise(
        X_list = toy$X, W_ref = toy$W_ref, blocks = lapply(toy$X, colnames), ncomp = 1L,
        sparsity = list(type = "c_vec", c_vec = c(a = 1, b = 1)),
        B = 6L, rng_streams = mb_rng_streams(6L, 84L), min_score_cor = 0.5
      )
    },
    "few accepted replicates for component\\(s\\) LC_01 \\(2 of B=6"
  )

  expect_identical(res$n_eff_by_component$n_eff, 2L)
  expect_equal(res$n_eff_by_component$fraction_effective, 2 / 6)
  expect_identical(res$components_below_effective_floor, "LC_01")
})

# Toy blocks whose centred values identify the resampled rows: `u` is the
# group index and `w` marks the second row of each two-row group.
grouped_toy_blocks = function(n_groups = 12L, seed = 85L) {
  set.seed(seed)
  n = 2L * n_groups
  a = cbind(u = rep(seq_len(n_groups), each = 2L), w = rep(c(0, 1), n_groups), a3 = rnorm(n))
  b = cbind(b1 = rnorm(n), b2 = rnorm(n))
  list(
    X = list(a = scale(a, scale = FALSE), b = scale(b, scale = FALSE)),
    W_ref = list(LC_01 = list(a = c(u = 0, w = 0, a3 = 1), b = c(b1 = 1, b2 = 0))),
    groups = sprintf("g%02d", rep(seq_len(n_groups), each = 2L))
  )
}

# Record block `a` of every replicate refit.
record_replicate_rows = function(rec) {
  function(X_blocks, ...) {
    rec$draws[[length(rec$draws) + 1L]] = X_blocks$a
    w_a = c(0, 0, 1)
    w_b = c(1, 0)
    list(
      W = list(list(w_a, w_b)),
      T_mat = cbind(X_blocks$a %*% w_a, X_blocks$b %*% w_b),
      converged = TRUE
    )
  }
}

# Whether both rows of every drawn group occur equally often. Centring keeps
# the ordering of `w` within a replicate, so `w > 0` marks the second rows.
draws_whole_groups = function(Xa) {
  second = factor(Xa[, "w"] > 0, levels = c(FALSE, TRUE))
  tab = table(round(Xa[, "u"], 8), second)
  all(tab[, "FALSE"] == tab[, "TRUE"])
}

test_that("replicates resample whole exchangeability groups", {
  toy = grouped_toy_blocks()
  rec = new.env()
  rec$draws = list()
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = record_replicate_rows(rec),
    .package = "mlr3mbspls"
  )
  selector = PipeOpMBsPLSBootstrapSelect$new()
  run = function(groups) {
    rec$draws = list()
    selector$.__enclos_env__$private$.bootstrap_align_and_summarise(
      X_list = toy$X, W_ref = toy$W_ref, blocks = lapply(toy$X, colnames), ncomp = 1L,
      sparsity = list(type = "c_vec", c_vec = c(a = 1, b = 1)),
      B = 8L, groups = groups, rng_streams = mb_rng_streams(8L, 86L), min_score_cor = 0
    )
    rec$draws
  }

  grouped = run(toy$groups)
  expect_length(grouped, 8L)
  expect_true(all(vapply(grouped, draws_whole_groups, logical(1))))
  # Each group enters with both rows, so the centred indicator is exactly +/- 0.5.
  expect_true(all(vapply(grouped, function(Xa) all(abs(abs(Xa[, "w"]) - 0.5) < 1e-12), logical(1))))
  # The draws are genuine resamples, not the training rows.
  expect_true(any(vapply(grouped, function(Xa) any(table(round(Xa[, "u"], 8)) != 2L), logical(1))))

  # A row bootstrap from the same streams splits groups.
  rows = run(NULL)
  expect_false(all(vapply(rows, draws_whole_groups, logical(1))))
})

test_that("the task's group role drives whole-group resampling in the PipeOp", {
  toy = grouped_toy_blocks()
  log_env = new.env(parent = emptyenv())
  log_env$mbspls_state = list(
    run_id = "group_role_draws",
    blocks = lapply(toy$X, colnames),
    weights = toy$W_ref,
    loadings = toy$W_ref,
    ncomp = 1L,
    X_train_blocks = toy$X,
    sparsity = list(type = "c_vec", c_vec = c(a = 1, b = 1))
  )
  log_env$mbspls_states = list(group_role_draws = log_env$mbspls_state)
  task = mlr3::TaskRegr$new(
    "group_role_draws",
    data.frame(
      LV1_a = toy$X$a[, "a3"], LV1_b = toy$X$b[, "b1"],
      participant = toy$groups, y = rnorm(nrow(toy$X$a))
    ),
    target = "y"
  )
  task$set_col_roles("participant", roles = "group")
  rec = new.env()
  rec$draws = list()
  testthat::local_mocked_bindings(
    cpp_mbspls_multi_lv = record_replicate_rows(rec),
    .package = "mlr3mbspls"
  )
  selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env, stability_only = TRUE, B = 8L, seed_bootstrap = 87L, min_score_cor = 0
  ))
  selector$train(list(task))

  expect_length(rec$draws, 8L)
  expect_true(all(vapply(rec$draws, draws_whole_groups, logical(1))))
  expect_identical(selector$state$bootstrap_group_source, "task_group_role")
  expect_identical(selector$state$n_exchangeability_units, 12L)
  expect_false(selector$state$few_exchangeability_units)
})

test_that("a group role with too few groups is refused or flagged", {
  set.seed(88L)
  n = 36L
  latent = rnorm(n)
  df = data.frame(
    x1 = latent + rnorm(n, 0, 0.4), x2 = latent + rnorm(n, 0, 0.4), x3 = rnorm(n),
    z1 = latent + rnorm(n, 0, 0.4), z2 = latent + rnorm(n, 0, 0.4), z3 = rnorm(n),
    y = rnorm(n)
  )
  blocks = list(b1 = c("x1", "x2", "x3"), b2 = c("z1", "z2", "z3"))
  select_with_sites = function(site) {
    task = mlr3::TaskRegr$new("few_groups", cbind(df, site = site), target = "y")
    task$set_col_roles("site", roles = "group")
    log_env = new.env(parent = emptyenv())
    upstream = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(
      ncomp = 1L, c_b1 = 1.4, c_b2 = 1.4, log_env = log_env, store_train_blocks = TRUE
    ))
    task_lv = upstream$train(list(task))[[1L]]
    selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
      log_env = log_env, stability_only = TRUE, B = 20L, seed_bootstrap = 89L
    ))
    list(selector = selector, task = task_lv)
  }

  # e.g. the training set of leave-one-site-out CV with two sites
  one = select_with_sites(rep("s1", n))
  expect_error(
    one$selector$train(list(one$task)),
    "only 1 exchangeability unit \\(groups of the task's group role 'site'\\).*Remove the group role"
  )

  three = select_with_sites(rep(c("s1", "s2", "s3"), each = n / 3))
  expect_match(
    collect_warnings(three$selector$train(list(three$task))),
    "only 3 exchangeability units \\(groups of the task's group role 'site'\\) can be resampled, i.e. at most 10 distinct",
    all = FALSE
  )
  expect_true(three$selector$state$few_exchangeability_units)
  expect_identical(three$selector$state$n_exchangeability_units, 3L)
})

test_that("the exchangeability unit check covers strata and distinct resamples", {
  check = getFromNamespace(".mb_bootstrap_unit_check", "mlr3mbspls")
  design = getFromNamespace(".mb_cluster_design", "mlr3mbspls")

  expect_error(
    check(design(c("g1", "g1", "g2", "g2"), c("s1", "s1", "s2", "s2")), 4L, 10L, "explicit", stratify_block = "grp"),
    "every stratum of `stratify_by_block` = 'grp' holds a single exchangeability unit.*drop `stratify_by_block`"
  )
  expect_error(check(NULL, 1L, 10L, "rows"), "only 1 exchangeability unit \\(training rows\\)")

  expect_no_warning({
    res = check(design(rep(1:12, each = 2L)), 24L, 500L, "explicit")
  })
  expect_identical(res$n_units, 12L)
  expect_false(res$few)
  expect_null(res$units_by_stratum)

  # Ten units in five strata of two give only 3^5 = 243 distinct samples.
  expect_warning(
    {
      res = check(design(1:10, rep(letters[1:5], each = 2L)), 10L, 500L, "rows", stratify_block = "grp")
    },
    "at most 243 distinct bootstrap samples for B = 500"
  )
  expect_true(res$few)
  expect_identical(res$units_by_stratum, c(a = 2L, b = 2L, c = 2L, d = 2L, e = 2L))

  # A singleton stratum next to a larger one is a valid design.
  expect_no_warning({
    res = check(design(1:30, c(rep("big", 29L), "single")), 30L, 500L, "rows", stratify_block = "grp")
  })
  expect_identical(res$units_by_stratum, c(big = 29L, single = 1L))
})

test_that("bootstrap = FALSE keeps the upstream LV columns at train and predict", {
  set.seed(90L)
  n = 30L
  latent = rnorm(n)
  df = data.frame(
    x1 = latent + rnorm(n, 0, 0.4), x2 = rnorm(n), x3 = rnorm(n),
    z1 = latent + rnorm(n, 0, 0.4), z2 = rnorm(n), z3 = rnorm(n),
    y = rnorm(n)
  )
  task = mlr3::TaskRegr$new("no_bootstrap", df, target = "y")
  blocks = list(b1 = c("x1", "x2", "x3"), b2 = c("z1", "z2", "z3"))

  graph = mbspls_graph(blocks = blocks, ncomp = 1L, bootstrap = FALSE)
  out_train = graph$train(task)[[1L]]
  out_pred = graph$predict(task)[[1L]]
  expect_setequal(out_train$feature_names, c("LV1_b1", "LV1_b2"))
  expect_identical(out_pred$feature_names, out_train$feature_names)
  expect_false(graph$pipeops$mbspls_bootstrap_select$state$bootstrap)

  learner = mlr3::as_learner(
    mbspls_graph(blocks = blocks, ncomp = 1L, bootstrap = FALSE) %>>%
      mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))
  )
  learner$train(task)
  expect_s3_class(learner$predict(task), "PredictionRegr")
})

test_that("stable predictions on new rows deflate with the stored stable loadings", {
  set.seed(92L)
  n = 80L
  f1 = rnorm(n)
  f2 = rnorm(n)
  b1 = cbind(
    x1 = f1 + rnorm(n, 0, 0.3), x2 = f1 + rnorm(n, 0, 0.3),
    x3 = f2 + rnorm(n, 0, 0.3), x4 = f2 + rnorm(n, 0, 0.3), x5 = rnorm(n)
  ) + 2
  b2 = cbind(
    z1 = f1 + rnorm(n, 0, 0.3), z2 = f1 + rnorm(n, 0, 0.3),
    z3 = f2 + rnorm(n, 0, 0.3), z4 = f2 + rnorm(n, 0, 0.3), z5 = rnorm(n)
  ) - 1
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))
  task = mlr3::TaskRegr$new("stable_new_rows", data.frame(b1, b2, y = rnorm(n)), target = "y")
  train_rows = 1:60
  new_rows = 61:80
  log_env = new.env(parent = emptyenv())
  upstream = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(
    ncomp = 2L, c_b1 = sqrt(2), c_b2 = sqrt(2), log_env = log_env,
    store_train_blocks = TRUE, append = TRUE, predict_weights = "raw"
  ))
  selector = PipeOpMBsPLSBootstrapSelect$new(param_vals = list(
    log_env = log_env, B = 20L, seed_bootstrap = 93L, selection_method = "frequency"
  ))
  selector$train(upstream$train(list(task$clone()$filter(train_rows))))
  expect_identical(selector$state$kept_components, 1:2)

  pred = selector$predict(upstream$predict(list(task$clone()$filter(new_rows))))[[1L]]

  # manual scores: centre with the upstream training means, score with the
  # stable weights and deflate with the stable training loadings
  st = selector$state
  X = lapply(names(blocks), function(b) {
    m = as.matrix(task$data(rows = new_rows, cols = blocks[[b]]))
    mu = st$center[[b]]
    sweep(m, 2L, mu[colnames(m)])
  })
  names(X) = names(blocks)
  expected = list()
  for (k in 1:2) {
    for (b in names(blocks)) {
      w = st$weights_stable[[k]][[b]][colnames(X[[b]])]
      expected[[paste0("LV", k, "_", b)]] = drop(X[[b]] %*% w)
    }
    for (b in names(blocks)) {
      t_kb = expected[[paste0("LV", k, "_", b)]]
      X[[b]] = X[[b]] - t_kb %o% st$loadings_stable[[k]][[b]][colnames(X[[b]])]
    }
  }
  kept = unlist(lapply(1:2, function(k) paste0("LV", k, "_", st$kept_blocks_per_comp[[k]])))
  expect_setequal(pred$feature_names, kept)
  expect_equal(as.data.frame(pred$data(cols = kept)), as.data.frame(expected[kept]))
})

test_that("parallel replicates report missing suggested packages", {
  toy = bootstrap_toy_blocks()
  testthat::local_mocked_bindings(
    .mbspls_has_namespace = function(pkg) pkg != "future.apply",
    .package = "mlr3mbspls"
  )
  expect_error(
    run_toy_bootstrap(toy, "block_sign", workers = 2L),
    "workers > 1 \\(or set workers = 1\\) requires the suggested package\\(s\\) 'future.apply'"
  )
})
