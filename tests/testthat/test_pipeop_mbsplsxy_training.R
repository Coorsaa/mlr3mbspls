test_that("PipeOpMBsPLSXY trains on classification tasks without levels shadowing", {
  set.seed(11)

  dt = data.table::data.table(
    x1 = rnorm(40),
    x2 = rnorm(40),
    x3 = rnorm(40),
    y = factor(sample(c("A", "B"), 40, replace = TRUE), levels = c("A", "B"))
  )

  task = mlr3::TaskClassif$new(id = "mbxy_train", backend = dt, target = "y")
  po = PipeOpMBsPLSXY$new(
    blocks = list(b1 = c("x1", "x2", "x3")),
    param_vals = list(ncomp = 1L, emit_y_scores = TRUE)
  )

  out = po$train(list(task))[[1L]]

  expect_s3_class(out, "Task")
  expect_true("LV1_b1" %in% out$feature_names)
  # Target scores are stored for inspection but never become features.
  expect_false("LV1_.Y" %in% out$feature_names)
  expect_true(is.matrix(po$state$scores_y))
  expect_equal(dim(po$state$scores_y), c(40L, 1L))
  expect_identical(colnames(po$state$scores_y), "LV1_.Y")
  expect_true(is.list(po$state$weights_x))
  expect_true(is.list(po$state$weights_y))
  expect_false(".Y_aux" %in% po$state$target_columns)
})

test_that("PipeOpMBsPLSXY rejects components beyond target rank", {
  set.seed(13)
  classification_task = mlr3::TaskClassif$new(
    id = "mbxy_rank_classif",
    backend = data.frame(
      x1 = rnorm(30L),
      x2 = rnorm(30L),
      y = factor(rep(c("A", "B"), 15L))
    ),
    target = "y"
  )
  regression_task = mlr3::TaskRegr$new(
    id = "mbxy_rank_regr",
    backend = data.frame(
      x1 = rnorm(30L),
      x2 = rnorm(30L),
      y = rnorm(30L)
    ),
    target = "y"
  )

  for (task in list(classification_task, regression_task)) {
    pipeop = PipeOpMBsPLSXY$new(
      blocks = list(block = c("x1", "x2")),
      param_vals = list(ncomp = 2L, y_rep = 3L)
    )
    expect_error(
      pipeop$train(list(task)),
      "effective rank 1"
    )
  }
})



test_that("PipeOpMBsPLSXY accepts rownamed c_matrix without explicit .target row", {
  set.seed(12)

  dt = data.table::data.table(
    x1 = rnorm(36),
    x2 = rnorm(36),
    x3 = rnorm(36),
    y = factor(sample(c("A", "B"), 36, replace = TRUE), levels = c("A", "B"))
  )

  task = mlr3::TaskClassif$new(id = "mbxy_cmat", backend = dt, target = "y")
  cm = matrix(1.5, nrow = 1L, ncol = 1L, dimnames = list("b1", "comp1"))

  po = PipeOpMBsPLSXY$new(
    blocks = list(b1 = c("x1", "x2", "x3")),
    param_vals = list(c_matrix = cm, c_target = 2, emit_y_scores = FALSE)
  )

  out = po$train(list(task))[[1L]]

  expect_s3_class(out, "Task")
  expect_true("LV1_b1" %in% out$feature_names)
  expect_equal(dim(po$state$c_matrix), c(2L, 1L))
  expect_equal(rownames(po$state$c_matrix), c("b1", ".target"))
  expect_lte(po$state$c_matrix[".target", 1L], sqrt(2))
})


test_that("PipeOpMBsPLSXY rejects malformed or infeasible c_matrix values", {
  set.seed(13)

  dt = data.table::data.table(
    x1 = rnorm(36),
    x2 = rnorm(36),
    y = factor(sample(c("A", "B"), 36, replace = TRUE))
  )
  task = mlr3::TaskClassif$new(id = "mbxy_bad_cmat", backend = dt, target = "y")
  blocks = list(b1 = c("x1", "x2"))

  duplicate_rows = matrix(
    1,
    nrow = 2L,
    ncol = 1L,
    dimnames = list(c("b1", "b1"), "LC1")
  )
  expect_error(
    PipeOpMBsPLSXY$new(
      blocks = blocks,
      param_vals = list(c_matrix = duplicate_rows)
    ),
    "row names must be unique"
  )

  duplicate_columns = matrix(
    1,
    nrow = 1L,
    ncol = 2L,
    dimnames = list("b1", c("LC1", "LC1"))
  )
  expect_error(
    PipeOpMBsPLSXY$new(
      blocks = blocks,
      param_vals = list(c_matrix = duplicate_columns)
    ),
    "column names must be unique"
  )

  bad_x_budget = matrix(
    2,
    nrow = 1L,
    ncol = 1L,
    dimnames = list("b1", "LC1")
  )
  po_x = PipeOpMBsPLSXY$new(
    blocks = blocks,
    param_vals = list(c_matrix = bad_x_budget)
  )
  expect_error(
    po_x$train(list(task)),
    "sparsity budget.*sqrt\\(p_block\\)"
  )

  bad_target_budget = matrix(
    c(1, 2),
    nrow = 2L,
    ncol = 1L,
    dimnames = list(c("b1", ".target"), "LC1")
  )
  po_target = PipeOpMBsPLSXY$new(
    blocks = blocks,
    param_vals = list(c_matrix = bad_target_budget)
  )
  expect_error(
    po_target$train(list(task)),
    "sparsity budget.*sqrt\\(p_block\\)"
  )
})


make_xy_class_task = function(n = 60L, seed = 21L, constant = character(0), n_classes = 3L) {
  set.seed(seed)
  y = factor(sample(LETTERS[seq_len(n_classes)], n, replace = TRUE))
  shift = as.numeric(y)
  dt = data.table::data.table(
    a1 = shift + rnorm(n), a2 = rnorm(n), a3 = rnorm(n),
    b1 = shift + rnorm(n), b2 = rnorm(n), b3 = rnorm(n),
    c1 = shift + rnorm(n), c2 = rnorm(n), c3 = rnorm(n),
    y = y
  )
  for (bn in constant) {
    cols = paste0(bn, 1:3)
    dt[, (cols) := 1]
  }
  list(
    task = mlr3::TaskClassif$new(id = "mbxy_layout", backend = dt, target = "y"),
    blocks = list(a = c("a1", "a2", "a3"), b = c("b1", "b2", "b3"), c = c("c1", "c2", "c3"))
  )
}


test_that("emit_y_scores never adds target-derived features and prediction works", {
  set.seed(31)
  dt = data.frame(x1 = rnorm(50), x2 = rnorm(50), z1 = rnorm(50), z2 = rnorm(50))
  dt$y = dt$x1 + dt$z1 + rnorm(50)
  task = mlr3::TaskRegr$new("mbxy_emit", dt, target = "y")
  blocks = list(a = c("x1", "x2"), b = c("z1", "z2"))

  gl = mlr3::as_learner(
    po("mbsplsxy", blocks = blocks, emit_y_scores = TRUE) %>>%
      mlr3pipelines::po("learner", mlr3::lrn("regr.featureless"))
  )
  gl$train(task)
  expect_setequal(gl$model$regr.featureless$train_task$feature_names, c("LV1_a", "LV1_b"))
  expect_no_error(gl$predict(task))
  expect_equal(dim(gl$model$mbsplsxy$scores_y), c(50L, 1L))

  gl_graph = mbsplsxy_graph_learner(
    learner = mlr3::lrn("regr.featureless"),
    task_type = "regr",
    blocks = blocks,
    ncomp = 1L,
    emit_y_scores = TRUE
  )
  gl_graph$train(task)
  expect_no_error(gl_graph$predict(task))
})


test_that("an unnamed c_matrix is interpreted against the declared XY layout", {
  d = make_xy_class_task(constant = "c")
  target_default = function(pipeop) min(5, sqrt(length(pipeop$state$target_columns)))

  po_x = PipeOpMBsPLSXY$new(blocks = d$blocks)
  po_x$param_set$values$c_matrix = matrix(c(1.2, 1.3, 1.4), ncol = 1L)
  po_x$train(list(d$task))
  cm = po_x$state$c_matrix
  expect_equal(rownames(cm), c("a", "b", ".target"))
  expect_equal(unname(cm[c("a", "b"), 1L]), c(1.2, 1.3))
  expect_equal(unname(cm[".target", 1L]), target_default(po_x))

  d_b = make_xy_class_task(constant = "b")
  po_b = PipeOpMBsPLSXY$new(blocks = d_b$blocks)
  po_b$param_set$values$c_matrix = matrix(c(1.2, 1.3, 1.4), ncol = 1L)
  po_b$train(list(d_b$task))
  expect_equal(rownames(po_b$state$c_matrix), c("a", "c", ".target"))
  expect_equal(unname(po_b$state$c_matrix[c("a", "c"), 1L]), c(1.2, 1.4))

  po_xt = PipeOpMBsPLSXY$new(blocks = d$blocks)
  po_xt$param_set$values$c_matrix = matrix(c(1.2, 1.3, 1.4, 1.5), ncol = 1L)
  po_xt$train(list(d$task))
  expect_equal(unname(po_xt$state$c_matrix[, 1L]), c(1.2, 1.3, 1.5))

  graph = mbsplsxy_graph(
    blocks = d$blocks,
    ncomp = 1L,
    c_matrix = matrix(c(1.2, 1.3, 1.4), ncol = 1L)
  )
  graph$train(d$task)
  expect_equal(unname(graph$pipeops$mbsplsxy$state$c_matrix[c("a", "b"), 1L]), c(1.2, 1.3))

  po_bad = PipeOpMBsPLSXY$new(blocks = d$blocks)
  po_bad$param_set$values$c_matrix = matrix(1.2, nrow = 2L, ncol = 1L)
  expect_error(po_bad$train(list(d$task)), "matched by position to the declared blocks")
})


test_that("MB-sPLS-XY rejects duplicated c_matrix rows set through param_set", {
  d = make_xy_class_task()
  po_dup = PipeOpMBsPLSXY$new(blocks = d$blocks)
  po_dup$param_set$values$c_matrix = matrix(
    c(1, 1, 1, 1.2, 1.3),
    ncol = 1L,
    dimnames = list(c("a", "b", "c", ".target", ".target"), "LC1")
  )
  expect_error(po_dup$train(list(d$task)), "row names must be unique")

  po_dup_x = PipeOpMBsPLSXY$new(blocks = d$blocks)
  po_dup_x$param_set$values$c_matrix = matrix(
    c(1, 1, 1, 1.5),
    ncol = 1L,
    dimnames = list(c("a", "b", "c", "a"), "LC1")
  )
  expect_error(po_dup_x$train(list(d$task)), "row names must be unique")
})


test_that("MB-sPLS-XY caps X budgets that exceed the retained width", {
  set.seed(41)
  n = 50L
  dt = data.table::data.table(
    a1 = rnorm(n), a2 = rnorm(n), a3 = rnorm(n), a4 = 1,
    b1 = rnorm(n), b2 = rnorm(n)
  )
  dt$y = dt$a1 + dt$b1 + rnorm(n)
  task = mlr3::TaskRegr$new("mbxy_capped", dt, target = "y")
  cm = matrix(c(2, 1.2), ncol = 1L, dimnames = list(c("a", "b"), "LC1"))
  pipeop = PipeOpMBsPLSXY$new(
    blocks = list(a = c("a1", "a2", "a3", "a4"), b = c("b1", "b2")),
    param_vals = list(c_matrix = cm)
  )
  expect_no_error(pipeop$train(list(task)))
  expect_equal(unname(pipeop$state$c_matrix["a", 1L]), sqrt(3))
})


test_that("MB-sPLS-XY stores train-time permutation p-values with their scope", {
  d = make_xy_class_task(n = 80L, seed = 22L, n_classes = 4L)
  set.seed(5)
  pipeop = PipeOpMBsPLSXY$new(
    blocks = d$blocks,
    param_vals = list(ncomp = 3L, permutation_test = TRUE, n_perm = 19L)
  )
  pipeop$train(list(d$task))
  st = pipeop$state

  expect_length(st$p_values, st$ncomp)
  expect_true(all(st$p_values > 0 & st$p_values <= 1))
  expect_match(st$p_value_scope, "not a full-pipeline permutation test")
  expect_length(st$converged, st$ncomp)
  expect_length(st$obj_vec, st$ncomp)

  sm = mbspls_model_summary(pipeop)
  expect_equal(sm$components$conditional_p_value, unname(st$p_values))
  expect_true(all(sm$components$p_value_scope == st$p_value_scope))
  expect_equal(sm$components$objective, unname(st$obj_vec))

  po_off = PipeOpMBsPLSXY$new(blocks = d$blocks, param_vals = list(ncomp = 1L))
  po_off$train(list(d$task))
  expect_null(po_off$state$p_value_scope)
  expect_true(all(is.na(po_off$state$p_values)))
})


test_that("MB-sPLS-XY always centres the target block", {
  d = make_xy_class_task()
  pipeop = PipeOpMBsPLSXY$new(blocks = d$blocks, param_vals = list(center_y = FALSE))
  expect_warning(pipeop$train(list(d$task)), "center_y = FALSE")
  reference = PipeOpMBsPLSXY$new(blocks = d$blocks)
  reference$train(list(d$task))
  expect_equal(pipeop$state$weights_x, reference$state$weights_x)
})
