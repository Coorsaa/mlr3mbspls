test_that("MB-sPLS payload helpers score the documented scalar summaries", {
  payload = list(
    mac_comp = c(0.5, 0.25),
    ev_comp = c(0.2, 0.8),
    ev_block = matrix(c(0.1, 0.3, 0.2, 0.4), nrow = 2L, byrow = TRUE),
    perf_metric = "mac",
    blocks = c("a", "b")
  )

  expect_equal(
    mbspls_measure_score_from_payload(payload, "mbspls.mac"),
    mean(payload$mac_comp)
  )
  expect_equal(
    mbspls_measure_score_from_payload(payload, "mbspls.ev"),
    mean(payload$ev_comp)
  )
  expect_equal(
    mbspls_measure_score_from_payload(payload, "mbspls.block_ev"),
    mean(as.numeric(payload$ev_block))
  )
  expect_equal(
    mbspls_measure_score_from_payload(payload, "mbspls.mac_evwt"),
    sum((payload$ev_comp / sum(payload$ev_comp)) * payload$mac_comp)
  )
})


test_that("MB-sPLS EV-weighted MAC uses positive-part EV weights and errors without positive EV", {
  payload_pos = list(
    mac_comp = c(0.2, 0.8, 0.5),
    ev_comp = c(-1, 2, -3),
    ev_block = matrix(0, nrow = 3L, ncol = 2L),
    perf_metric = "mac",
    blocks = c("a", "b")
  )
  expect_equal(
    mbspls_measure_score_from_payload(payload_pos, "mbspls.mac_evwt"),
    0.8
  )

  payload_all_nonpos = list(
    mac_comp = c(0.1, 0.9),
    ev_comp = c(-2, -1),
    ev_block = matrix(0, nrow = 2L, ncol = 2L),
    perf_metric = "mac",
    blocks = c("a", "b")
  )
  expect_error(
    mbspls_measure_score_from_payload(payload_all_nonpos, "mbspls.mac_evwt"),
    "no component has positive finite prediction-side explained variance"
  )

  err = tryCatch(
    mbspls_measure_score_from_payload(payload_all_nonpos, "mbspls.mac_evwt"),
    error = function(e) e
  )
  expect_s3_class(err, "mbspls_undefined_measure_score")

  diag = mbspls_measure_score_diagnostics(payload_all_nonpos, "mbspls.mac_evwt")
  expect_true(is.na(diag$score))
  expect_false(diag$defined)
  expect_identical(diag$reason, "nonpositive_ev")
  expect_match(diag$message, "undefined")

  diag_ok = mbspls_measure_score_diagnostics(payload_pos, "mbspls.mac_evwt")
  expect_true(diag_ok$defined)
  expect_equal(diag_ok$score, 0.8)
})


test_that("MB-sPLS MAC normalises Frobenius scores by the number of block pairs", {
  payload = list(
    mac_comp = c(sqrt(3)),
    ev_comp = 1,
    ev_block = matrix(1, nrow = 1L, ncol = 3L),
    perf_metric = "frobenius",
    blocks = c("a", "b", "c")
  )

  expect_equal(
    mbspls_measure_score_from_payload(payload, "mbspls.mac"),
    1
  )
})


test_that("MB-sPCA payload helper computes mean EV and rejects unsupported ids", {
  payload = list(
    ev_comp = c(0.2, 0.4, -0.1),
    ev_block = matrix(c(0.1, 0.2, 0.3), nrow = 3L),
    blocks = c("a")
  )

  expect_equal(
    mbspca_measure_score_from_payload(payload, "mbspca.mean_ev"),
    mean(payload$ev_comp)
  )
  expect_error(
    mbspca_measure_score_from_payload(payload, "mbspls.mac"),
    "Unsupported MB-sPCA measure"
  )
})


test_that("registered measures are task-type agnostic technical measures", {
  mids = c("mbspls.mac_evwt", "mbspls.mac", "mbspls.ev", "mbspls.block_ev", "mbspca.mean_ev")
  ms = lapply(mids, mlr3::msr)

  for (m in ms) {
    expect_true(is.na(m$task_type))
    expect_true("requires_learner" %in% m$properties)
    expect_true("requires_no_prediction" %in% m$properties)
    expect_true(is.na(m$predict_type))
  }
})
