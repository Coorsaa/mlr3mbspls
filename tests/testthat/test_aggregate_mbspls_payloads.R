test_that("aggregate_mbspls_payloads aggregates minimal payload lists", {
  payloads = list(
    list(
      mac_comp = c(0.50, 0.40),
      ev_comp = c(0.30, 0.20),
      ev_block = matrix(
        c(0.15, 0.10,
          0.12, 0.08),
        nrow = 2,
        dimnames = list(NULL, c("b1", "b2"))
      ),
      T_mat = matrix(0, nrow = 10, ncol = 4),
      blocks = c("b1", "b2"),
      perf_metric = "mac"
    ),
    list(
      mac_comp = c(0.60, 0.45),
      ev_comp = c(0.32, 0.22),
      ev_block = matrix(
        c(0.16, 0.11,
          0.13, 0.09),
        nrow = 2,
        dimnames = list(NULL, c("b1", "b2"))
      ),
      T_mat = matrix(0, nrow = 12, ncol = 4),
      blocks = c("b1", "b2"),
      perf_metric = "mac"
    )
  )

  agg = aggregate_mbspls_payloads(payloads)

  expect_type(agg, "list")
  expect_true(all(c("summary", "fold_table") %in% names(agg)))
  expect_true(is.list(agg$summary))

  s = agg$summary
  expect_true(all(c(
    "mac_mean", "mac_sd", "ev_comp_mean", "ev_block_mean", "p_combined",
    "p_combination_method", "p_combination_scope", "blocks"
  ) %in% names(s)))
  expect_identical(s$p_combination_method, "none")
  expect_true(all(is.na(s$p_combined)))

  expect_true(is.numeric(s$mac_mean))
  expect_length(s$mac_mean, 2L)

  expect_true(is.numeric(s$ev_comp_mean))
  expect_length(s$ev_comp_mean, 2L)

  expect_true(is.matrix(s$ev_block_mean))
  expect_equal(dim(s$ev_block_mean), c(2L, 2L))
  expect_setequal(colnames(s$ev_block_mean), c("b1", "b2"))
})


test_that("aggregate_mbspls_payloads combines p-values once per fold/component", {
  payloads = list(
    list(
      mac_comp = c(0.50),
      ev_comp = c(0.30),
      ev_block = matrix(c(0.15, 0.10), nrow = 1, dimnames = list(NULL, c("b1", "b2"))),
      val_test_p = c(0.10),
      T_mat = matrix(0, nrow = 10, ncol = 2),
      blocks = c("b1", "b2"),
      perf_metric = "mac"
    ),
    list(
      mac_comp = c(0.60),
      ev_comp = c(0.32),
      ev_block = matrix(c(0.16, 0.11), nrow = 1, dimnames = list(NULL, c("b1", "b2"))),
      val_test_p = c(0.20),
      T_mat = matrix(0, nrow = 12, ncol = 2),
      blocks = c("b1", "b2"),
      perf_metric = "mac"
    )
  )

  expect_error(
    aggregate_mbspls_payloads(payloads, p_method = "stouffer"),
    "not combined by default"
  )
  expect_warning(
    aggregate_mbspls_payloads(
      payloads,
      p_method = "stouffer",
      allow_p_combination = TRUE
    ),
    "exploratory"
  )
  agg = suppressWarnings(aggregate_mbspls_payloads(
    payloads,
    p_method = "stouffer",
    allow_p_combination = TRUE
  ))
  expected = 1 - stats::pnorm((sqrt(10) * stats::qnorm(0.90) + sqrt(12) * stats::qnorm(0.80)) /
    sqrt(10 + 12))

  expect_equal(unname(agg$summary$p_combined[[1L]]), expected, tolerance = 1e-12)
})


test_that("exploratory p-value aggregation preserves extreme tails", {
  payload = list(
    mac_comp = 0.5, ev_comp = 0.3, val_test_p = 1e-30,
    T_mat = matrix(0, 10, 2), blocks = c("a", "b")
  )
  result = suppressWarnings(aggregate_mbspls_payloads(
    list(payload), p_method = "stouffer", allow_p_combination = TRUE
  ))
  expect_equal(log(unname(result$summary$p_combined)), log(1e-30), tolerance = 1e-12)
  payload$val_test_p = 0
  result = suppressWarnings(aggregate_mbspls_payloads(
    list(payload), p_method = "fisher", allow_p_combination = TRUE
  ))
  expect_equal(unname(result$summary$p_combined), 0)
  payload$val_test_p = 1.1
  expect_error(suppressWarnings(aggregate_mbspls_payloads(
    list(payload), p_method = "stouffer", allow_p_combination = TRUE
  )), "finite values in")
})

test_that("aggregate_mbspls_payloads handles monotone enforcement with one component", {
  payloads = list(
    list(
      mac_comp = c(0.50),
      ev_comp = c(0.30),
      ev_block = matrix(c(0.15, 0.10), nrow = 1, dimnames = list(NULL, c("b1", "b2"))),
      val_test_p = c(0.10),
      T_mat = matrix(0, nrow = 10, ncol = 2),
      blocks = c("b1", "b2"),
      perf_metric = "mac"
    )
  )

  expect_warning(
    aggregate_mbspls_payloads(
      payloads,
      p_method = "stouffer",
      allow_p_combination = TRUE,
      enforce_monotone = TRUE
    ),
    "exploratory"
  )
})
