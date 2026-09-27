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

encoded_block_task = function(n = 40L, seed = 11L) {
  set.seed(seed)
  data = data.table::data.table(
    a1 = stats::rnorm(n), a2 = stats::rnorm(n), a3 = stats::rnorm(n),
    b1 = stats::rnorm(n), b2 = stats::rnorm(n), b3 = stats::rnorm(n),
    grp = factor(rep_len(c("lo", "mid", "hi"), n), levels = c("lo", "mid", "hi"))
  )
  data$a1 = data$a1 + as.integer(data$grp)
  data$b1 = data$b1 + as.integer(data$grp)
  mlr3cluster::TaskClust$new("encoded_blocks", data)
}

test_that("TunerSeqMBsPLS resolves encoded block columns exactly as PipeOpMBsPLS", {
  task = encoded_block_task()
  for (blocks in list(
    list(A = c("a1", "a2", "a3", "grp"), B = c("b1", "b2", "b3")),
    list(A = "grp", B = c("b1", "b2", "b3"))
  )) {
    gl = mbspls_graph_learner(
      learner = mlr3::lrn("clust.kmeans", centers = 2L), blocks = blocks, ncomp = 1L,
      bootstrap = FALSE, bootstrap_selection = FALSE, val_test = "none"
    )
    instance = mlr3tuning::ti(
      task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr("mbspls.mac"), terminator = bbotk::trm("evals", n_evals = 1L)
    )
    tuner = TunerSeqMBsPLS$new(budget = 2L, resampling = mlr3::rsmp("cv", folds = 2L),
      early_stopping = FALSE)
    expect_no_error(tuner$optimize(instance))
    resolved = tuner$diagnostics$blocks
    expect_true(all(c("grp.mid", "grp.hi") %in% resolved$A))
    expect_false("grp" %in% resolved$A)

    gl$param_set$values$mbspls.c_matrix = instance$result_learner_param_vals$c_matrix
    gl$train(task)
    expect_identical(resolved, gl$model$mbspls$blocks)
  }
})

test_that("TunerSeqMBsPCA resolves encoded block columns with the shared resolver", {
  task = encoded_block_task()
  for (blocks in list(
    list(A = c("a1", "a2", "a3", "grp"), B = c("b1", "b2", "b3")),
    list(A = "grp", B = c("b1", "b2", "b3"))
  )) {
    pre = mlr3pipelines::po("encode", method = "treatment") %>>% mlr3pipelines::po("scale")
    gl = mlr3::as_learner(pre %>>% PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L)) %>>%
      mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
    instance = mlr3tuning::ti(
      task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr("mbspca.mean_ev"), terminator = bbotk::trm("evals", n_evals = 1L)
    )
    tuner = TunerSeqMBsPCA$new(budget = 2L, resampling = mlr3::rsmp("cv", folds = 2L),
      early_stopping = FALSE)
    expect_no_error(tuner$optimize(instance))
    columns = pre$clone(deep = TRUE)$train(task$clone(deep = TRUE))[[1L]]$feature_names
    expect_identical(tuner$diagnostics$blocks, mb_resolve_block_columns(columns, blocks))
    expect_true(all(c("grp.mid", "grp.hi") %in% tuner$diagnostics$blocks$A))
  }
})

test_that("sequential tuners drop blocks without usable columns like their PipeOps", {
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 17L)
  blocks = task$block_features()
  data = task$data(cols = unlist(blocks))
  data$const = 1
  task = mlr3cluster::TaskClust$new("empty_block", data)
  blocks$empty = "const"
  for (method in c("pls", "pca")) {
    component = if (method == "pls") {
      PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L, append = FALSE))
    } else {
      PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L))
    }
    gl = mlr3::as_learner(component %>>%
      mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
    instance = mlr3tuning::ti(
      task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr(if (method == "pls") "mbspls.mac" else "mbspca.mean_ev"),
      terminator = bbotk::trm("evals", n_evals = 1L)
    )
    constructor = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
    expect_warning(
      constructor$new(budget = 1L, resampling = mlr3::rsmp("cv", folds = 2L),
        early_stopping = FALSE)$optimize(instance),
      "dropped.*: empty"
    )
    c_matrix = instance$result_learner_param_vals$c_matrix
    expect_identical(rownames(c_matrix), c("block_a", "block_b", "block_c"))
    gl$param_set$values[[paste0(component$id, ".c_matrix")]] = c_matrix
    expect_no_error(gl$train(task))
  }
})

test_that("sequential tuners train preprocessing only on the full task and inner training folds", {
  task = task_multiblock_synthetic(task_type = "clust", n = 36L, seed = 19L)
  blocks = task$block_features()
  task$select(unique(unlist(blocks)))
  rs = mlr3::rsmp("cv", folds = 3L)
  rs$instantiate(task)
  for (method in c("pls", "pca")) {
    recorder = PipeOpRecordTrainRows$new()
    component = if (method == "pls") {
      PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L, append = FALSE))
    } else {
      PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L))
    }
    gl = mlr3::as_learner(recorder %>>% mlr3pipelines::po("scale") %>>% component %>>%
      mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
    instance = mlr3tuning::ti(
      task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr(if (method == "pls") "mbspls.mac" else "mbspca.mean_ev"),
      terminator = bbotk::trm("evals", n_evals = 1L)
    )
    constructor = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
    constructor$new(budget = 1L, resampling = rs, early_stopping = FALSE)$optimize(instance)
    # One fit on the full task to resolve the final blocks, then one per inner
    # training fold; validation rows never reach a training step.
    expected = c(list(sort(task$row_ids)), lapply(seq_len(rs$iters), function(f) sort(rs$train_set(f))))
    expect_identical(recorder$record$train, expected)
  }
})

test_that("sequential tuners resolve blocks from the PipeOp's feature columns only", {
  set.seed(23L)
  n = 40L
  data = data.table::data.table(
    a.x1 = stats::rnorm(n), a.x2 = stats::rnorm(n), a.x3 = stats::rnorm(n),
    b1 = stats::rnorm(n), b2 = stats::rnorm(n), b3 = stats::rnorm(n)
  )
  data$a.y = data$a.x1 + data$b1 + stats::rnorm(n)
  task = mlr3::TaskRegr$new("prefixed_target", data, target = "a.y")
  blocks = list(A = "a", B = c("b1", "b2", "b3"))

  # The declared prefix `a` must not claim the target `a.y`, and columns
  # outside `affect_columns` are never block members.
  for (selector in list(NULL, mlr3pipelines::selector_invert(mlr3pipelines::selector_name("a.x3")))) {
    po_mb = PipeOpMBsPLS$new(blocks = blocks, param_vals = list(ncomp = 1L, append = FALSE))
    po_mb$param_set$values$affect_columns = selector
    gl = mlr3::as_learner(po_mb %>>% mlr3pipelines::po("learner", mlr3::lrn("regr.featureless")))
    instance = mlr3tuning::ti(
      task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr("mbspls.mac"), terminator = bbotk::trm("evals", n_evals = 1L)
    )
    tuner = TunerSeqMBsPLS$new(budget = 1L, resampling = mlr3::rsmp("cv", folds = 2L), early_stopping = FALSE)
    tuner$optimize(instance)
    expected_a = if (is.null(selector)) c("a.x1", "a.x2", "a.x3") else c("a.x1", "a.x2")
    expect_identical(tuner$diagnostics$blocks$A, expected_a)

    gl$param_set$values$mbspls.c_matrix = instance$result_learner_param_vals$c_matrix
    gl$train(task)
    expect_identical(tuner$diagnostics$blocks, gl$model$mbspls$blocks)
  }

  gl = mlr3::as_learner(PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 1L)) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("regr.featureless")))
  instance = mlr3tuning::ti(
    task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("mbspca.mean_ev"), terminator = bbotk::trm("evals", n_evals = 1L)
  )
  tuner = TunerSeqMBsPCA$new(budget = 1L, resampling = mlr3::rsmp("cv", folds = 2L), early_stopping = FALSE)
  tuner$optimize(instance)
  expect_identical(tuner$diagnostics$blocks, mb_resolve_block_columns(task$feature_names, blocks))
  expect_false("a.y" %in% unlist(tuner$diagnostics$blocks))
})

test_that("sequential tuners apply the PipeOps' component rank guard before searching", {
  set.seed(29L)
  n = 40L
  task = mlr3cluster::TaskClust$new("binary_block", data.table::data.table(
    a1 = stats::rnorm(n), a2 = stats::rnorm(n), b1 = stats::rnorm(n), b2 = stats::rnorm(n),
    sex = factor(rep_len(c("f", "m"), n))
  ))
  # One binary factor gives a single dummy column, which supports one component.
  blocks = list(A = c("a1", "a2"), B = c("b1", "b2"), S = "sex")
  gl = mbspls_graph_learner(
    learner = mlr3::lrn("clust.kmeans", centers = 2L), blocks = blocks, ncomp = 2L,
    bootstrap = FALSE, bootstrap_selection = FALSE, val_test = "none"
  )
  instance = mlr3tuning::ti(
    task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("mbspls.mac"), terminator = bbotk::trm("evals", n_evals = 1L)
  )
  expect_error(
    TunerSeqMBsPLS$new(budget = 2L, resampling = mlr3::rsmp("cv", folds = 2L), early_stopping = FALSE)$optimize(instance),
    "PipeOpMBsPLS \\(full tuning task\\) requested 2 components, but effective block rank is lower for: S \\(1\\)"
  )

  pre = mlr3pipelines::po("encode", method = "treatment") %>>% mlr3pipelines::po("scale")
  gl = mlr3::as_learner(pre %>>% PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2L)) %>>%
    mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L)))
  instance = mlr3tuning::ti(
    task = task, learner = gl, resampling = mlr3::rsmp("holdout"),
    measure = mlr3::msr("mbspca.mean_ev"), terminator = bbotk::trm("evals", n_evals = 1L)
  )
  expect_error(
    TunerSeqMBsPCA$new(budget = 2L, resampling = mlr3::rsmp("cv", folds = 2L), early_stopping = FALSE)$optimize(instance),
    "PipeOpMBsPCA \\(full tuning task\\) requested 2 components, but effective block rank is lower for: S \\(1\\)"
  )
})
