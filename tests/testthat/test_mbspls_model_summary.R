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
  expect_true(all(c("component", "objective", "p_value", "ev_comp") %in% names(sm$components)))
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
