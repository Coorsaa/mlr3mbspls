test_that("PipeOpMBsPLS - trains and predicts with per-block c parameters", {
  set.seed(42)

  n = 80
  p_clin = 6
  p_gen = 20

  clinical = matrix(rnorm(n * p_clin), nrow = n, ncol = p_clin)
  colnames(clinical) = paste0("c", seq_len(p_clin))
  genomic = matrix(rnorm(n * p_gen), nrow = n, ncol = p_gen)
  colnames(genomic) = paste0("g", seq_len(p_gen))
  age = rnorm(n)
  y = rnorm(n)

  # Add a non-block feature (age) to verify append=TRUE behaviour.
  df = data.frame(clinical, genomic, age = age, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(
    clinical = colnames(clinical),
    genomic = colnames(genomic)
  )

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 2L,
      c_clinical = sqrt(5L),
      c_genomic = sqrt(6L),
      append = FALSE
    )
  )

  out_train = po$train(list(task))[[1]]
  lv_cols = c(
    "LV1_clinical", "LV1_genomic",
    "LV2_clinical", "LV2_genomic"
  )

  expect_s3_class(out_train, "Task")
  expect_setequal(out_train$feature_names, lv_cols)

  st = po$state
  expect_true(is.list(st))
  expect_equal(st$ncomp, 2L)
  expect_setequal(names(st$blocks), names(blocks))
  expect_length(st$weights, 2L)
  expect_setequal(names(st$weights[[1]]), names(blocks))
  expect_true(all(vapply(st$weights[[1]], is.numeric, logical(1))))

  task_new = task$clone(deep = TRUE)
  task_new$filter(1:10)

  out_pred = po$predict(list(task_new))[[1]]
  expect_s3_class(out_pred, "Task")
  expect_equal(out_pred$nrow, 10)
  expect_setequal(out_pred$feature_names, lv_cols)
})


test_that("PipeOpMBsPLS - append = TRUE keeps raw features and adds LVs", {
  set.seed(123)

  n = 60
  p_clin = 4
  p_gen = 10

  clinical = matrix(rnorm(n * p_clin), nrow = n, ncol = p_clin)
  colnames(clinical) = paste0("c", seq_len(p_clin))
  genomic = matrix(rnorm(n * p_gen), nrow = n, ncol = p_gen)
  colnames(genomic) = paste0("g", seq_len(p_gen))
  age = rnorm(n)
  y = rnorm(n)

  df = data.frame(clinical, genomic, age = age, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(
    clinical = colnames(clinical),
    genomic = colnames(genomic)
  )

  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_clinical = 2L,
      c_genomic = 2L,
      append = TRUE
    )
  )

  out_train = po$train(list(task))[[1]]
  lv_cols = c("LV1_clinical", "LV1_genomic")

  expect_true(all(lv_cols %in% out_train$feature_names))
  expect_true("age" %in% out_train$feature_names)
  expect_true(all(c(blocks$clinical, blocks$genomic) %in% out_train$feature_names))
})


test_that("PipeOpMBsPLS - c_matrix path works and validates dimensions", {
  set.seed(1)

  n = 50
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

  Cmat = matrix(sqrt(3), nrow = 2, ncol = 2)
  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 2L,
      c_matrix = Cmat,
      append = FALSE
    )
  )
  out_train = po$train(list(task))[[1]]

  expect_true(is.matrix(po$state$c_matrix))
  expect_equal(dim(po$state$c_matrix), c(2L, 2L))
  expect_true(all(c("LV1_b1", "LV1_b2", "LV2_b1", "LV2_b2") %in% out_train$feature_names))

  # wrong dimensions should error early
  expect_error(
    PipeOpMBsPLS$new(
      blocks = blocks,
      param_vals = list(
        ncomp = 2L,
        c_matrix = matrix(1, nrow = 1, ncol = 2),
        append = FALSE
      )
    ),
    sprintf("c_matrix must have %s rows \\(blocks\\); got 1", length(blocks))
  )

  excessive = matrix(100, nrow = 2L, ncol = 1L)
  po_excessive = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(c_matrix = excessive, append = FALSE)
  )
  expect_error(
    po_excessive$train(list(task)),
    "sparsity budget.*sqrt\\(p_block\\)"
  )
})


test_that("PipeOpMBsPLS - writes expected payloads to log_env", {
  set.seed(7)

  n = 40
  p1 = 4
  p2 = 6

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())
  po = PipeOpMBsPLS$new(
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

  po$train(list(task))
  expect_true(is.list(log_env$mbspls_state))
  expect_true(all(c("blocks", "weights", "loadings", "T_mat_train") %in% names(log_env$mbspls_state)))

  po$predict(list(task))
  expect_true(is.list(log_env$last))
  expect_true(all(c("mac_comp", "ev_block", "ev_comp", "T_mat") %in% names(log_env$last)))
})

test_that("PipeOpMBsPLS - prediction-side bootstrap logs descriptive uncertainty", {
  set.seed(71)

  n = 44
  p1 = 5
  p2 = 6

  b1 = matrix(rnorm(n * p1), nrow = n, ncol = p1)
  colnames(b1) = paste0("x", seq_len(p1))
  b2 = matrix(rnorm(n * p2), nrow = n, ncol = p2)
  colnames(b2) = paste0("z", seq_len(p2))
  y = rnorm(n)

  df = data.frame(b1, b2, y = y)
  task = mlr3::TaskRegr$new(id = "mb_boot_val", backend = df, target = "y")
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  log_env = new.env(parent = emptyenv())
  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_b1 = 2L,
      c_b2 = 2L,
      log_env = log_env,
      val_test = "bootstrap",
      val_test_n = 20L,
      val_test_alpha = 0.05,
      append = FALSE
    )
  )

  po$train(list(task))
  po$predict(list(task))

  expect_true(is.list(log_env$last))
  expect_true("val_bootstrap" %in% names(log_env$last))
  expect_true(is.data.frame(log_env$last$val_bootstrap))
  expect_equal(nrow(log_env$last$val_bootstrap), 1L)
  expect_true(all(c("estimate", "bootstrap_mean", "bias", "standard_error",
    "conf_low", "conf_high", "replicates_effective", "p_value_note") %in%
    names(log_env$last$val_bootstrap)))
  expect_true(is.na(log_env$last$val_bootstrap$p_value[[1L]]))
  expect_false("val_test_p" %in% names(log_env$last))
})


test_that("PipeOpMBsPLS accepts rownamed c_matrix entries for retained blocks after a block drops out", {
  task0 = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 111L)
  blocks = task0$block_features()

  dt = data.table::as.data.table(task0$data(cols = task0$feature_names))
  dt[, (blocks[[3L]]) := 1]
  task = TaskMultiBlock(dt, blocks = blocks, task_type = "clust", id = "mbspls_drop")

  cm = matrix(sqrt(2), nrow = 3L, ncol = 1L, dimnames = list(names(blocks), NULL))
  po = PipeOpMBsPLS$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 1L,
      c_matrix = cm,
      append = FALSE
    )
  )

  expect_no_error(po$train(list(task)))
  expect_equal(names(po$state$blocks), names(blocks)[1:2])
  expect_equal(dim(po$state$c_matrix), c(2L, 1L))
})


test_that("PipeOpMBsPLS rejects non-finite blocks and impossible component counts", {
  non_finite_task = mlr3::TaskRegr$new(
    "mbspls_non_finite",
    data.frame(
      x1 = c(1, 2, NA, 4),
      x2 = c(4, 3, 2, 1),
      z1 = c(1, 3, 2, 4),
      z2 = c(2, 4, 1, 3),
      y = 1:4
    ),
    target = "y"
  )
  non_finite = PipeOpMBsPLS$new(
    blocks = list(x = c("x1", "x2"), z = c("z1", "z2"))
  )
  expect_error(non_finite$train(list(non_finite_task)), "finite")

  rank_task = mlr3::TaskRegr$new(
    "mbspls_rank",
    data.frame(x = 1:8, z1 = c(1:7, 9), z2 = c(8:2, 0), y = 1:8),
    target = "y"
  )
  rank_limited = PipeOpMBsPLS$new(
    blocks = list(x = "x", z = c("z1", "z2")),
    param_vals = list(ncomp = 2L)
  )
  expect_error(rank_limited$train(list(rank_task)), "effective block rank")
})


make_mbspls_signal_blocks = function(n = 70L, n_blocks = 2L, seed = 1L) {
  set.seed(seed)
  l1 = rnorm(n)
  l2 = rnorm(n)
  mats = lapply(seq_len(n_blocks), function(b) {
    m = cbind(
      l1 + rnorm(n, sd = 0.4),
      l2 + rnorm(n, sd = 0.6),
      matrix(rnorm(n * 3L), n)
    )
    colnames(m) = paste0("b", b, "_", seq_len(ncol(m)))
    m
  })
  names(mats) = paste0("b", seq_len(n_blocks))
  list(
    task = mlr3::TaskRegr$new(
      id = "mbspls_signal",
      backend = data.frame(do.call(cbind, mats), y = rnorm(n)),
      target = "y"
    ),
    blocks = lapply(mats, colnames)
  )
}

publish_stable_weights = function(log_env, state, W_stable) {
  st_env = log_env$mbspls_states[[state$run_id]]
  st_env$weights_stable = W_stable
  st_env$loadings_stable = state$loadings
  st_env$selection_method = "ci"
  log_env_store_state(log_env, st_env, warn_overwrite = FALSE)
}


test_that("prediction-side diagnostics skip components with fewer than two informative blocks", {
  d = make_mbspls_signal_blocks()
  log_env = new.env(parent = emptyenv())
  pipeop = PipeOpMBsPLS$new(
    blocks = d$blocks,
    param_vals = list(
      ncomp = 2L, c_b1 = 1.5, c_b2 = 1.5, log_env = log_env,
      val_test_n = 19L, seed_validation = 11L
    )
  )
  pipeop$train(list(d$task))
  st = pipeop$state

  # Stability selection may zero a whole block of a later component.
  W_stable = st$weights
  W_stable$LC_02$b2[] = 0
  publish_stable_weights(log_env, st, W_stable)

  pipeop$param_set$values$val_test = "permutation"
  expect_no_error(pipeop$predict(list(d$task)))
  payload = log_env$last
  expect_identical(payload$weights_source, "stable_ci")
  expect_true(is.finite(payload$val_test_p[["LC_01"]]))
  expect_true(payload$val_test_p[["LC_01"]] > 0 && payload$val_test_p[["LC_01"]] <= 1)
  expect_true(is.na(payload$val_test_p[["LC_02"]]))
  expect_identical(payload$val_test_status[["LC_01"]], "computed")
  expect_match(payload$val_test_status[["LC_02"]], "fewer than two blocks")

  # LC_01 keeps its own RNG stream, so skipping LC_02 does not change it.
  pipeop$param_set$values$predict_weights = "raw"
  pipeop$predict(list(d$task))
  expect_identical(log_env$last$val_test_p[["LC_01"]], payload$val_test_p[["LC_01"]])
  pipeop$param_set$values$predict_weights = "auto"

  pipeop$param_set$values$val_test = "bootstrap"
  expect_no_error(pipeop$predict(list(d$task)))
  boot = log_env$last$val_bootstrap
  expect_equal(nrow(boot), 2L)
  expect_true(is.finite(boot$estimate[[1L]]))
  expect_true(is.na(boot$estimate[[2L]]))
  expect_identical(boot$replicates_effective[[2L]], 0L)
  expect_match(boot$p_value_note[[2L]], "fewer than two blocks")
  expect_match(log_env$last$val_test_status[["LC_02"]], "fewer than two blocks")
})


test_that("prediction-side diagnostics still run when two informative blocks remain", {
  d = make_mbspls_signal_blocks(n_blocks = 3L, seed = 2L)
  log_env = new.env(parent = emptyenv())
  pipeop = PipeOpMBsPLS$new(
    blocks = d$blocks,
    param_vals = list(
      ncomp = 2L, c_b1 = 1.5, c_b2 = 1.5, c_b3 = 1.5, log_env = log_env,
      val_test = "permutation", val_test_n = 19L, seed_validation = 3L
    )
  )
  pipeop$train(list(d$task))
  W_stable = pipeop$state$weights
  W_stable$LC_02$b3[] = 0
  publish_stable_weights(log_env, pipeop$state, W_stable)

  pipeop$predict(list(d$task))
  expect_true(all(is.finite(log_env$last$val_test_p)))
  expect_true(all(log_env$last$val_test_status == "computed"))
})


test_that("c_matrix budgets are bounded by structural widths and capped at retained widths", {
  set.seed(9)
  n = 50L
  a = cbind(matrix(rnorm(n * 3L), n), 1)
  colnames(a) = paste0("a", 1:4)
  b = matrix(rnorm(n * 3L), n)
  colnames(b) = paste0("b", 1:3)
  task = mlr3::TaskRegr$new("mbspls_capped", data.frame(a, b, y = rnorm(n)), target = "y")
  blocks = list(a = colnames(a), b = colnames(b))

  # 2 = sqrt(4) is admissible for the declared block a, whose constant column
  # is removed in these training rows.
  cm = matrix(c(2, sqrt(3)), nrow = 2L, dimnames = list(c("a", "b"), "LC1"))
  po_cm = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(c_matrix = cm))
  expect_no_error(po_cm$train(list(task)))
  expect_equal(unname(po_cm$state$c_matrix["a", 1L]), sqrt(3))
  expect_null(attr(po_cm$state$c_matrix, "capped"))

  po_vec = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(c_a = 2, c_b = sqrt(3)))
  po_vec$train(list(task))
  expect_equal(po_cm$state$weights, po_vec$state$weights, tolerance = 1e-8)

  too_large = matrix(c(2.1, 1), nrow = 2L, dimnames = list(c("a", "b"), "LC1"))
  po_bad = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(c_matrix = too_large))
  expect_error(po_bad$train(list(task)), "sparsity budget.*sqrt\\(p_block\\)")
})


test_that("c_matrix set through param_set rejects ambiguous rows and uses the declared layout", {
  d = make_mbspls_signal_blocks(n_blocks = 3L, seed = 4L)
  pipeop = PipeOpMBsPLS$new(blocks = d$blocks)

  pipeop$param_set$values$c_matrix = matrix(
    c(1, 1, 1, 2),
    nrow = 4L,
    dimnames = list(c("b1", "b2", "b3", "b1"), "LC1")
  )
  expect_error(pipeop$train(list(d$task)), "row names must be unique")

  pipeop$param_set$values$c_matrix = matrix(1, nrow = 3L, dimnames = list(c("b1", NA, "b3"), "LC1"))
  expect_error(pipeop$train(list(d$task)), "row names must be unique and non-empty")

  # An unnamed matrix is matched to the declared blocks even when one drops out.
  dt = data.table::as.data.table(d$task$data())
  dt[, (d$blocks$b2) := 1]
  task_drop = mlr3::TaskRegr$new("mbspls_declared", dt, target = "y")
  pipeop$param_set$values$c_matrix = matrix(c(1.2, 1.3, 1.4), nrow = 3L)
  pipeop$train(list(task_drop))
  expect_equal(rownames(pipeop$state$c_matrix), c("b1", "b3"))
  expect_equal(unname(pipeop$state$c_matrix[, 1L]), c(1.2, 1.4))
})


test_that("PipeOpMBsPLS stores named training EV and solver convergence", {
  d = make_mbspls_signal_blocks(n_blocks = 3L, seed = 5L)
  pipeop = PipeOpMBsPLS$new(
    blocks = d$blocks,
    param_vals = list(ncomp = 2L, c_b1 = 1.5, c_b2 = 1.5, c_b3 = 1.5)
  )
  pipeop$train(list(d$task))
  st = pipeop$state
  comps = c("LC_01", "LC_02")

  expect_null(dim(st$ev_comp))
  expect_named(st$ev_comp, comps)
  expect_identical(dimnames(st$ev_block), list(comps, names(d$blocks)))
  # ev_comp is SS-weighted across the (centred) blocks, not the row sum.
  sst = vapply(st$blocks, function(cols) {
    x = as.matrix(d$task$data(cols = cols))
    sum(sweep(x, 2L, colMeans(x))^2)
  }, numeric(1L))
  expect_equal(unname(st$ev_comp), as.numeric(st$ev_block %*% (sst / sum(sst))), tolerance = 1e-8)

  expect_named(st$converged, comps)
  expect_type(st$converged, "logical")
  expect_named(st$iterations, comps)
  expect_type(st$iterations, "integer")
  expect_true(all(st$iterations >= 1L))

  expect_warning(
    .mb_warn_nonconverged(c(LC_01 = TRUE, LC_02 = FALSE), "[x] MB-sPLS", 600L),
    "did not converge within 600 iterations for component\\(s\\) LC_02"
  )
  expect_no_warning(.mb_warn_nonconverged(c(LC_01 = TRUE), "[x] MB-sPLS", 600L))
})
