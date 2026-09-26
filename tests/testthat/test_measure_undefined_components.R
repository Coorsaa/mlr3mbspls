undefined_component_payload = function(mac_comp, ev_comp) {
  list(
    mac_comp = mac_comp,
    ev_comp = ev_comp,
    ev_block = matrix(0.1, nrow = length(mac_comp), ncol = 2L),
    perf_metric = "mac",
    blocks = c("a", "b")
  )
}


test_that("MAC measures score undefined components as zero association", {
  payload = undefined_component_payload(c(0.6, NaN), c(0.3, 0.3))

  expect_equal(mbspls_measure_score_from_payload(payload, "mbspls.mac_evwt"), 0.3)
  expect_equal(mbspls_measure_score_from_payload(payload, "mbspls.mac"), 0.3)

  single = undefined_component_payload(NaN, 0.2)
  expect_identical(mbspls_measure_score_from_payload(single, "mbspls.mac_evwt"), 0)
  expect_identical(mbspls_measure_score_from_payload(single, "mbspls.mac"), 0)
})


test_that("a component collapsing to one block does not outscore a weak defined component", {
  degenerate = undefined_component_payload(c(0.6, NaN), c(0.3, 0.3))
  weak = undefined_component_payload(c(0.6, 0.1), c(0.3, 0.3))

  for (id in c("mbspls.mac_evwt", "mbspls.mac")) {
    expect_lt(
      mbspls_measure_score_from_payload(degenerate, id),
      mbspls_measure_score_from_payload(weak, id)
    )
  }
})


test_that("score diagnostics report undefined components", {
  payload = undefined_component_payload(c(0.6, NaN), c(0.3, 0.3))
  for (id in c("mbspls.mac_evwt", "mbspls.mac", "mbspls.ev")) {
    diag = mbspls_measure_score_diagnostics(payload, id)
    expect_true(diag$defined)
    expect_identical(diag$n_undefined_components, 1L)
  }

  defined = undefined_component_payload(c(0.6, 0.4), c(0.3, 0.3))
  expect_identical(mbspls_measure_score_diagnostics(defined, "mbspls.mac")$n_undefined_components, 0L)

  missing = mbspls_measure_score_diagnostics(NULL, "mbspls.mac")
  expect_false(missing$defined)
  expect_identical(missing$n_undefined_components, NA_integer_)

  nonpositive = undefined_component_payload(c(NaN, 0.4), c(-0.1, -0.2))
  diag = mbspls_measure_score_diagnostics(nonpositive, "mbspls.mac_evwt")
  expect_false(diag$defined)
  expect_identical(diag$reason, "nonpositive_ev")
  expect_identical(diag$n_undefined_components, 1L)
})
