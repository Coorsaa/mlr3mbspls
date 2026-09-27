test_that("PipeOpMBsPLS weights do not depend on seed_train", {
  task = task_multiblock_synthetic(n = 60L, seed = 2L)
  blocks = task$block_features()
  fit_with_seed = function(seed) {
    po = PipeOpMBsPLS$new(
      blocks = blocks,
      param_vals = list(
        ncomp = 2L,
        c_block_a = 1.5,
        c_block_b = 1.5,
        c_block_c = 1.5,
        seed_train = seed
      )
    )
    po$train(list(task))
    po$state
  }
  first = fit_with_seed(1L)
  second = fit_with_seed(2L)
  expect_identical(second$weights, first$weights)
  expect_identical(second$obj_vec, first$obj_vec)
})

test_that("observed permutation-test statistic does not depend on analysis_seed", {
  task = task_multiblock_synthetic(n = 60L, seed = 3L)
  blocks = lapply(task$block_features(), function(cols) {
    as.matrix(task$data(cols = cols))
  })
  statistic = vapply(c(1L, 11L), function(analysis_seed) {
    mbspls_permutation_test(
      blocks,
      ncomp = 1L,
      statistic = "global_lc1",
      n_perm = 9L,
      seed = 5L,
      analysis_seed = analysis_seed
    )$statistic
  }, numeric(1L))
  expect_identical(statistic[[2L]], statistic[[1L]])
})
