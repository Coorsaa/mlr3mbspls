test_that("mbspls_model_summary formats MB-sPLS states", {
  blocks = list(block_a = c("a1", "a2"), block_b = c("b1"))
  po = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L))
  po$state = list(
    blocks = blocks,
    weights = list(LC_01 = list(
      block_a = stats::setNames(c(1, 0), blocks$block_a),
      block_b = stats::setNames(0.5, blocks$block_b)
    )),
    loadings = list(LC_01 = list(
      block_a = stats::setNames(c(0.4, 0.1), blocks$block_a),
      block_b = stats::setNames(0.3, blocks$block_b)
    )),
    ev_block = matrix(c(0.6, 0.2), nrow = 1L, dimnames = list("LC_01", names(blocks))),
    ev_comp = c(LC_01 = 0.8),
    obj_vec = c(LC_01 = 0.9),
    p_values = c(LC_01 = 0.01),
    performance_metric = "mac",
    correlation_method = "pearson",
    run_id = "run_mbspls"
  )

  sm = mbspls_model_summary(po)

  expect_equal(sm$overview$model[[1L]], "mbspls")
  expect_equal(sm$overview$n_components[[1L]], 1L)
  expect_true(all(c("component", "objective", "conditional_p_value", "p_value_scope", "ev_comp") %in%
    names(sm$components)))
  expect_false("p_value" %in% names(sm$components))
  expect_equal(sm$components$conditional_p_value, 0.01)
  # Without a stored scope the qualifier is missing rather than invented.
  expect_true(is.na(sm$components$p_value_scope))
  expect_true(all(c("component", "block", "feature", "weight", "loading", "selected") %in% names(sm$weights)))
  expect_true(any(sm$weights$selected))
})


test_that("mbspls_model_summary formats MB-sPCA states", {
  blocks = list(block_a = c("a1", "a2"), block_b = c("b1", "b2"))
  po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L))
  po$state = list(
    blocks = blocks,
    weights = list(PC1 = list(
      block_a = stats::setNames(c(1, 0), blocks$block_a),
      block_b = stats::setNames(c(0.5, 0), blocks$block_b)
    )),
    loadings = list(PC1 = list(
      block_a = stats::setNames(c(0.3, 0.1), blocks$block_a),
      block_b = stats::setNames(c(0.2, 0.05), blocks$block_b)
    )),
    ev_block = matrix(c(0.4, 0.3), nrow = 1L, dimnames = list("PC1", names(blocks))),
    ev_comp = c(PC1 = 0.7),
    run_id = "run_mbspca"
  )

  sm = mbspls_model_summary(po)

  expect_equal(sm$overview$model[[1L]], "mbspca")
  expect_true(all(c("component", "ev_comp") %in% names(sm$components)))
  expect_true(all(c("component", "block", "n_features", "n_selected", "ev_block") %in% names(sm$blocks)))
})


test_that("mb_named_ev_block aligns named matrices and rejects unknown names", {
  ev_block = matrix(
    c(1, 2, 3, 4),
    nrow = 2L,
    byrow = TRUE,
    dimnames = list(c("LC_02", "LC_01"), c("block_b", "block_a"))
  )

  aligned = mb_named_ev_block(
    ev_block = ev_block,
    component_names = c("LC_01", "LC_02"),
    block_names = c("block_a", "block_b")
  )

  expect_equal(unname(aligned["LC_01", "block_a"]), 4)
  expect_equal(unname(aligned["LC_01", "block_b"]), 3)
  expect_equal(unname(aligned["LC_02", "block_a"]), 2)
  expect_equal(unname(aligned["LC_02", "block_b"]), 1)

  bad_ev_block = matrix(
    1,
    nrow = 1L,
    dimnames = list("LC_01", "block_c")
  )

  expect_error(
    mb_named_ev_block(
      ev_block = bad_ev_block,
      component_names = c("LC_01"),
      block_names = c("block_a")
    ),
    "unknown block names"
  )
})


test_that("mbspls_model_summary formats MB-sPLS-XY states including target weights", {
  blocks = list(block_a = c("a1", "a2"), block_b = c("b1"))
  po = PipeOpMBsPLSXY$new(blocks = blocks, param_vals = list(ncomp = 1L))
  po$state = list(
    blocks_x = blocks,
    target_columns = c(".Y_case", ".Y_control"),
    ncomp = 1L,
    weights_x = list(LC_01 = list(
      block_a = stats::setNames(c(1, 0), blocks$block_a),
      block_b = stats::setNames(0.5, blocks$block_b)
    )),
    loadings_x = list(LC_01 = list(
      block_a = stats::setNames(c(0.4, 0.1), blocks$block_a),
      block_b = stats::setNames(0.2, blocks$block_b)
    )),
    weights_y = list(LC_01 = stats::setNames(c(0.8, -0.8), c(".Y_case", ".Y_control"))),
    loadings_y = list(LC_01 = stats::setNames(c(0.5, -0.5), c(".Y_case", ".Y_control"))),
    performance_metric = "mac",
    correlation_method = "pearson",
    emit_y_scores = TRUE
  )

  sm = mbspls_model_summary(po)

  expect_equal(sm$overview$model[[1L]], "mbsplsxy")
  expect_true(any(sm$weights$block == ".target"))
  expect_true(all(c("component", "block", "feature", "weight", "loading", "selected") %in% names(sm$weights)))
})


test_that("mbspls_model_summary labels train-time p-values with their scope", {
  blocks = list(block_a = c("a1", "a2"), block_b = c("b1"))
  po = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L))
  base_state = list(
    blocks = blocks,
    weights = list(LC_01 = list(
      block_a = stats::setNames(c(1, 0), blocks$block_a),
      block_b = stats::setNames(0.5, blocks$block_b)
    )),
    loadings = list(LC_01 = list(
      block_a = stats::setNames(c(0.4, 0.1), blocks$block_a),
      block_b = stats::setNames(0.3, blocks$block_b)
    )),
    ev_comp = c(LC_01 = 0.8),
    obj_vec = c(LC_01 = 0.9),
    performance_metric = "mac"
  )

  po$state = c(base_state, list(
    p_values = c(LC_01 = 0.01),
    p_value_scope = "Conditional component-wise diagnostic."
  ))
  sm = mbspls_model_summary(po)
  expect_equal(sm$components$conditional_p_value, 0.01)
  expect_identical(sm$components$p_value_scope, "Conditional component-wise diagnostic.")

  po$state = c(base_state, list(p_values = c(LC_01 = NA_real_)))
  sm = mbspls_model_summary(po)
  expect_true(is.na(sm$components$conditional_p_value))
  expect_true(is.na(sm$components$p_value_scope))

  po$state = base_state
  sm = mbspls_model_summary(po)
  expect_true(is.na(sm$components$conditional_p_value))
  expect_true(is.na(sm$components$p_value_scope))
})


test_that("mbspls_model_summary reports the MB-sPCA cross-block diagnostic", {
  task = task_multiblock_synthetic(task_type = "clust", n = 36L, seed = 108L)
  po = PipeOpMBsPCA$new(
    blocks = task$block_features(),
    param_vals = list(ncomp = 1L, permutation_test = TRUE, n_perm = 19L)
  )
  po$train(list(task))
  sm = mbspls_model_summary(po)
  expect_equal(sm$components$conditional_p_value, unname(po$state$p_values))
  expect_match(sm$components$p_value_scope, "cross-block")
  expect_true(all(sm$blocks$n_selected > 0L))
})


test_that("mbspls_model_summary counts selected features of older unnamed MB-sPCA states", {
  blocks = list(block_a = c("a1", "a2"), block_b = c("b1", "b2"))
  po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L))
  po$state = list(
    blocks = blocks,
    weights = list(PC1 = list(
      stats::setNames(c(1, 0), blocks$block_a),
      stats::setNames(c(0.6, 0.8), blocks$block_b)
    )),
    loadings = list(PC1 = list(
      stats::setNames(c(0.3, 0.1), blocks$block_a),
      stats::setNames(c(0.2, 0.05), blocks$block_b)
    )),
    ev_comp = c(PC1 = 0.7)
  )
  sm = mbspls_model_summary(po)
  expect_identical(sm$blocks$n_selected, c(1L, 2L))
  expect_equal(sm$weights$weight, c(1, 0, 0.6, 0.8))
})
