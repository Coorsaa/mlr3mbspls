test_that("mb_task_overview summarises packaged multiblock tasks", {
  task = task_multiblock_synthetic(task_type = "classif", n = 30L, seed = 1L)

  qc = mb_task_overview(task)

  expect_type(qc, "list")
  expect_true(all(c("overview", "blocks", "target", "target_distribution", "issues") %in% names(qc)))
  expect_equal(qc$overview$n_blocks[[1L]], 3L)
  expect_equal(qc$overview$n_rows[[1L]], task$nrow)
  expect_setequal(qc$blocks$block, c("block_a", "block_b", "block_c"))
  expect_equal(qc$target$target[[1L]], "subtype")
  expect_true(all(c("level", "n", "proportion") %in% names(qc$target_distribution)))

  qc_method = task$overview()
  expect_equal(qc_method$overview$n_blocks[[1L]], qc$overview$n_blocks[[1L]])
})


test_that("mb_task_overview reports missingness and constant numeric features", {
  dt = data.table::data.table(
    site = factor(c("A", "A", "B", "B", "A", "B")),
    clin_age = c(40, 40, 40, 40, 40, 40),
    clin_bmi = c(21, NA, 25, 26, NA, 29),
    omics_1 = c(1, 2, 3, 4, 5, 6),
    omics_2 = c(NA, 1, 0, 1, 0, 1),
    y = factor(c("a", "a", "a", "b", "b", "b"))
  )
  blocks = list(clin = c("clin_age", "clin_bmi"), omics = c("omics_1", "omics_2"))
  task = TaskMultiBlock(dt, blocks = blocks, target = "y", task_type = "classif")

  qc = mb_task_overview(task)

  expect_true(any(qc$issues$issue == "constant_numeric_features"))
  expect_true(any(qc$issues$issue == "missing_values"))
  expect_true(any(qc$blocks$block == "clin" & qc$blocks$n_constant_numeric > 0L))
  expect_true(any(qc$blocks$block == "clin" & qc$blocks$pct_missing_cells > 0))
})
