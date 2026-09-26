batchtools_nested_cv = function(task, rs_outer, reg_dir) {
  gl = mbspls_graph_learner(
    learner = mlr3::lrn("clust.kmeans", centers = 2L),
    task = task,
    ncomp = 1L,
    bootstrap = FALSE,
    bootstrap_selection = FALSE,
    val_test = "none"
  )
  mbspls_nested_cv_batchtools(
    task = task, graphlearner = gl, rs_outer = rs_outer, rs_inner = mlr3::rsmp("holdout"),
    ncomp = 1L, tuner_budget = 1L, tuning_early_stop = FALSE, n_perm_tuning = 1L,
    reg_dir = reg_dir,
    cluster_function = batchtools::makeClusterFunctionsInteractive(external = FALSE)
  )
}

test_that("collect_mbspls_nested_cv reports unfinished and failed outer folds", {
  testthat::skip_if_not_installed("batchtools")
  task = task_multiblock_synthetic(task_type = "clust", n = 45L, seed = 36L)
  ids = task$row_ids
  # Split 2 has too few analysis rows for the inner holdout and therefore errors.
  rs_outer = mlr3::rsmp("custom")
  rs_outer$instantiate(task,
    train_sets = list(ids[1:30], ids[1:4], ids[16:45]),
    test_sets = list(ids[31:45], ids[31:45], ids[1:15])
  )
  reg_dir = tempfile("mbspls_nested_cv_reg_")
  on.exit(unlink(reg_dir, recursive = TRUE), add = TRUE)
  out = suppressMessages(batchtools_nested_cv(task, rs_outer, reg_dir))

  expect_named(out, c("ids", "reg"))
  expect_s3_class(out$reg, "Registry")
  expect_identical(out$ids$job.id, 1:3)
  expect_error(collect_mbspls_nested_cv(reg = out$reg),
    "3 of 3 outer fold jobs did not complete")

  suppressMessages(suppressWarnings(batchtools::submitJobs(ids = 1:2, reg = out$reg)))
  expect_error(
    collect_mbspls_nested_cv(reg = out$reg),
    "2 of 3 outer fold jobs did not complete \\(errors: 1, expired: 0, not finished: 1; outer splits 2, 3\\).*allow_partial"
  )

  res = NULL
  expect_warning(
    {
      res = collect_mbspls_nested_cv(reg = out, allow_partial = TRUE)
    },
    "2 of 3 outer fold jobs did not complete"
  )
  expect_identical(res$results$split, 1:3)
  expect_true(is.finite(res$results$measure_test[1L]))
  expect_true(all(is.na(res$results$measure_test[2:3])))
  expect_match(res$results$measure_test_status[2L], "^Outer fold job failed: ")
  expect_match(res$results$measure_test_status[3L], "has not finished")
  primary = res$summary_table[1L, ]
  expect_identical(c(primary$n_total, primary$n_defined, primary$n_failed), c(3L, 1L, 2L))
  expect_length(res$c_mats, 3L)
  expect_true(is.matrix(res$c_mats[[1L]]) && is.null(res$c_mats[[2L]]))
  expect_identical(is.na(res$inner_scores), c(FALSE, TRUE, TRUE))

  # Explicitly requested finished jobs are collected without complaint.
  res_done = collect_mbspls_nested_cv(ids = 1L, reg = out$reg)
  expect_identical(res_done$results$split, 1L)
  expect_identical(res_done$summary_table$n_total[1L], 1L)
  expect_identical(collect_mbspls_nested_cv(ids = out$ids[1L], reg = out$reg)$results, res_done$results)
  # Requested ids that are not in the registry are reported, not dropped.
  expect_error(collect_mbspls_nested_cv(ids = c(1L, 7L, 9L), reg = out$reg),
    "Job id\\(s\\) not found in the registry: 7, 9")
})

test_that("mbspls_nested_cv_batchtools reproduces the in-process result schema", {
  testthat::skip_if_not_installed("batchtools")
  task = task_multiblock_synthetic(task_type = "clust", n = 40L, seed = 37L)
  rs_outer = mlr3::rsmp("cv", folds = 2L)
  rs_outer$instantiate(task)
  reg_dir = tempfile("mbspls_nested_cv_reg_")
  on.exit(unlink(reg_dir, recursive = TRUE), add = TRUE)
  out = suppressMessages(batchtools_nested_cv(task, rs_outer, reg_dir))
  suppressMessages(batchtools::submitJobs(reg = out$reg))
  res = NULL
  expect_no_warning({
    res = collect_mbspls_nested_cv(reg = out$reg)
  })
  expect_identical(res$results$split, 1:2)
  expect_true(all(is.finite(res$results$measure_test)))
  expect_length(res$inner_scores, 2L)
  expect_identical(unique(res$summary_table$n_total), 2L)

  gl = mbspls_graph_learner(
    learner = mlr3::lrn("clust.kmeans", centers = 2L), task = task, ncomp = 1L,
    bootstrap = FALSE, bootstrap_selection = FALSE, val_test = "none"
  )
  direct = mbspls_nested_cv(
    task = task, graphlearner = gl, rs_outer = rs_outer, rs_inner = mlr3::rsmp("holdout"),
    ncomp = 1L, tuner_budget = 1L, tuning_early_stop = FALSE, n_perm_tuning = 1L
  )
  expect_identical(names(res$results), names(direct$results))
  expect_identical(names(res$summary_table), names(direct$summary_table))
  expect_identical(res$summary_table$metric, direct$summary_table$metric)
})

test_that("mbspls_nested_cv_batchtools requires batchtools before touching the file system", {
  testthat::local_mocked_bindings(
    .mbspls_has_namespace = function(pkg) !identical(pkg, "batchtools"),
    .package = "mlr3mbspls"
  )
  reg_dir = tempfile("mbspls_nested_cv_reg_")
  on.exit(unlink(reg_dir, recursive = TRUE), add = TRUE)
  task = task_multiblock_synthetic(task_type = "clust", n = 20L, seed = 38L)
  expect_error(
    mbspls_nested_cv_batchtools(
      task = task, graphlearner = NULL, rs_outer = mlr3::rsmp("holdout"),
      rs_inner = mlr3::rsmp("holdout"), ncomp = 1L, tuner_budget = 1L, reg_dir = reg_dir
    ),
    "mbspls_nested_cv_batchtools\\(\\) requires the suggested package\\(s\\) 'batchtools'"
  )
  expect_false(dir.exists(reg_dir))
  expect_error(collect_mbspls_nested_cv(reg = NULL),
    "collect_mbspls_nested_cv\\(\\) requires the suggested package\\(s\\) 'batchtools'")
})
