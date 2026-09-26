test_that("sequential tuners preserve single and multiple preprocessing nodes", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 311L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  for (method in c("pls", "pca")) {
    for (n_pre in 1:2) {
      component = if (method == "pls") {
        PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L))
      } else {
        PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L))
      }
      pre = mlr3pipelines::as_graph(mlr3pipelines::po("scale"))
      if (n_pre == 2L) {
        pre = mlr3pipelines::po("colapply", applicator = function(x) x + 100) %>>% pre
      }
      gl = mlr3::as_learner(pre %>>% component %>>%
        mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
      extracted = mb_preprocessing_graph(gl$graph, component$id)
      expect_identical(extracted$ids(), pre$ids())
      expect_equal(extracted$edges, pre$edges)
      transformed = extracted$train(task$clone(deep = TRUE))[[1L]]$data()
      expected = pre$clone(deep = TRUE)$train(task$clone(deep = TRUE))[[1L]]$data()
      expect_equal(transformed, expected)
      expect_true(all(abs(colMeans(as.matrix(transformed))) < 1e-10))

      measure = if (method == "pls") "mbspls.mac" else "mbspca.mean_ev"
      tuner = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
      instance = mlr3tuning::ti(
        task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
        measure = mlr3::msr(measure), terminator = bbotk::trm("evals", n_evals = 1L)
      )
      expect_no_error(tuner$new(budget = 1L,
        resampling = mlr3::rsmp("cv", folds = 2L),
        early_stopping = FALSE)$optimize(instance))
      expect_true(is.finite(instance$result_y))
      expect_null(gl$model)
    }
  }
})

test_that("resampling guards reject row overlap, stale rows and group leakage", {
  task = mlr3cluster::TaskClust$new("grouped", data.table::data.table(
    x = seq_len(8L), participant = rep(letters[1:4], each = 2L)
  ))
  task$set_col_roles("participant", "group")
  expect_error(mb_assert_resampling_split(task, 1:4, 4:8), "must be disjoint")
  expect_error(mb_assert_resampling_split(task, 1:4, 9:10), "current task")
  expect_error(mb_assert_resampling_split(task, c(1, 3, 5, 7), c(2, 4, 6, 8)),
    "overlap|both", ignore.case = TRUE)
  expect_no_error(mb_assert_resampling_split(task, 1:4, 5:8))
})
