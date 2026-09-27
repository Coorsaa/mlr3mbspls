test_that("PipeOpMBsPCA stores training scores from the sequentially deflated fit", {
  task = task_multiblock_synthetic(task_type = "clust", n = 36L, seed = 101L)
  blocks = task$block_features()

  po = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(
      ncomp = 2L,
      permutation_test = FALSE
    )
  )

  out_train = po$train(list(task))[[1L]]
  st = po$state

  X_cur = lapply(names(st$blocks), function(bn) {
    M = as.matrix(task$data(cols = st$blocks[[bn]]))
    storage.mode(M) = "double"
    # Scores refer to blocks centred by the stored training means.
    sweep(M, 2L, st$center[[bn]][colnames(M)])
  })

  T_list = vector("list", st$ncomp)
  for (k in seq_len(st$ncomp)) {
    Wk = st$weights[[k]]
    Pk = st$loadings[[k]]
    Tk = do.call(cbind, lapply(seq_along(st$blocks), function(b) {
      X_cur[[b]] %*% unname(Wk[[b]])
    }))
    colnames(Tk) = paste0("PC", k, "_", names(st$blocks))
    T_list[[k]] = Tk

    if (k < st$ncomp) {
      for (b in seq_along(st$blocks)) {
        X_cur[[b]] = X_cur[[b]] - tcrossprod(Tk[, b], unname(Pk[[b]]))
      }
    }
  }

  T_expected = do.call(cbind, T_list)
  expect_equal(st$T_mat, T_expected, tolerance = 1e-8)
  expect_equal(
    as.matrix(out_train$data(cols = colnames(T_expected))),
    T_expected,
    tolerance = 1e-8
  )
})


test_that("PipeOpMBsPCA validates c_matrix rows against retained blocks", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 102L)
  blocks = task$block_features()

  cm_named = matrix(2, nrow = 1L, ncol = 2L, dimnames = list("wrong_block", NULL))
  po_named = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(ncomp = 2L, c_matrix = cm_named)
  )
  expect_error(
    po_named$train(list(task)),
    "rows must match either all declared or all retained blocks"
  )

  cm_plain = matrix(2, nrow = 2L, ncol = 2L)
  po_plain = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(ncomp = 2L, c_matrix = cm_plain)
  )
  expect_error(
    po_plain$train(list(task)),
    "must have 3 rows"
  )
})


test_that("PipeOpMBsPCA accepts a retained-block c_matrix after a block drops out", {
  task0 = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 103L)
  blocks = task0$block_features()

  dt = data.table::as.data.table(task0$data(cols = task0$feature_names))
  dt[, (blocks[[3L]]) := 1]
  task = TaskMultiBlock(dt, blocks = blocks, task_type = "clust", id = "mbspca_drop")

  cm = matrix(2, nrow = 2L, ncol = 1L, dimnames = list(names(blocks)[1:2], NULL))
  po = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(ncomp = 1L, c_matrix = cm)
  )

  expect_no_error(po$train(list(task)))
  expect_equal(names(po$state$blocks), names(blocks)[1:2])
  expect_equal(po$state$ncomp, 1L)

  # An unnamed matrix is matched by position to the declared blocks, so the
  # budget of the dropped block is discarded instead of shifting the others.
  cm_declared = matrix(c(1.5, 2, 1.2), nrow = 3L, ncol = 1L)
  po_declared = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(c_matrix = cm_declared)
  )
  po_declared$train(list(task))
  expect_equal(rownames(po_declared$state$c_matrix), names(blocks)[1:2])
  expect_equal(unname(po_declared$state$c_matrix[, 1L]), c(1.5, 2))

  po_retained = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(c_matrix = matrix(2, nrow = 2L, ncol = 1L))
  )
  expect_error(po_retained$train(list(task)), "matched by position to the declared blocks")
})


test_that("PipeOpMBsPCA rejects infeasible and ambiguous c_matrix values", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 104L)
  blocks = task$block_features()

  # Real block names with one repeated: a conflicting budget must not be
  # dropped silently.
  duplicate_rows = matrix(
    c(1, 1, 1, 1.4),
    nrow = 4L,
    ncol = 1L,
    dimnames = list(c(names(blocks), names(blocks)[[1L]]), "PC1")
  )
  po_duplicate = PipeOpMBsPCA$new(blocks = blocks)
  po_duplicate$param_set$values$c_matrix = duplicate_rows
  expect_error(po_duplicate$train(list(task)), "row names must be unique")

  wrong_names = matrix(
    1,
    nrow = 3L,
    ncol = 1L,
    dimnames = list(c("clinical", "genomic", "proteomic"), "PC1")
  )
  po_wrong = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(c_matrix = wrong_names)
  )
  expect_error(
    po_wrong$train(list(task)),
    "rows must match either all declared or all retained blocks"
  )

  excessive = matrix(100, nrow = 3L, ncol = 1L)
  po_excessive = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(c_matrix = excessive)
  )
  expect_error(
    po_excessive$train(list(task)),
    "sparsity budget.*sqrt\\(p_block\\)"
  )
})


test_that("PipeOpMBsPCA stops after a non-significant first diagnostic", {
  task = task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 105L)
  po = PipeOpMBsPCA$new(
    blocks = task$block_features(),
    param_vals = list(
      ncomp = 3L,
      permutation_test = TRUE,
      n_perm = 2L,
      perm_alpha = 0
    )
  )

  po$train(list(task))

  expect_identical(po$state$ncomp, 1L)
  expect_length(po$state$p_values, 1L)
  expect_gt(po$state$p_values[[1L]], 0)
})


test_that("PipeOpMBsPCA rejects non-finite blocks and impossible ranks", {
  non_finite = mlr3::TaskUnsupervised$new(
    id = "mbspca_non_finite",
    backend = data.frame(
      x1 = c(1, 2, NA, 4),
      x2 = c(4, 3, 2, 1),
      z1 = c(1, 3, 2, 4),
      z2 = c(2, 4, 1, 3)
    )
  )
  po_non_finite = PipeOpMBsPCA$new(
    blocks = list(x = c("x1", "x2"), z = c("z1", "z2"))
  )
  expect_error(po_non_finite$train(list(non_finite)), "finite")

  rank_limited = mlr3::TaskUnsupervised$new(
    id = "mbspca_rank",
    backend = data.frame(x = 1:8, z1 = c(1:7, 9), z2 = c(8:2, 0))
  )
  po_rank = PipeOpMBsPCA$new(
    blocks = list(x = "x", z = c("z1", "z2")),
    param_vals = list(ncomp = 2L)
  )
  expect_error(po_rank$train(list(rank_limited)), "effective block rank")
})

test_that("PCA default sparsity remains valid after constant features are removed", {
  task = mlr3::TaskUnsupervised$new(
    id = "pca_filtered_dimension",
    backend = data.frame(x = 1:8, constant = 1, y = c(1:7, 9))
  )
  po = PipeOpMBsPCA$new(blocks = list(a = c("x", "constant"), b = "y"))
  expect_no_error(po$train(list(task)))
  expect_identical(po$state$blocks$a, "x")
  expect_equal(abs(unname(po$state$weights[[1L]][[1L]])), 1)
})


test_that("PipeOpMBsPCA skips the cross-block permutation diagnostic for fewer than two blocks", {
  set.seed(51)
  n = 40L
  f1 = rnorm(n)
  f2 = rnorm(n)
  x = cbind(f1 + rnorm(n, sd = 0.2), f1 + rnorm(n, sd = 0.2), f2 + rnorm(n, sd = 0.3), f2 + rnorm(n, sd = 0.3))
  colnames(x) = paste0("x", 1:4)
  single = mlr3::TaskUnsupervised$new("mbspca_single", backend = data.frame(x))

  po_single = PipeOpMBsPCA$new(
    blocks = list(a = colnames(x)),
    param_vals = list(ncomp = 2L, permutation_test = TRUE, n_perm = 19L)
  )
  expect_warning(po_single$train(list(single)), "at least two usable blocks")
  expect_identical(po_single$state$ncomp, 2L)
  expect_true(all(is.na(po_single$state$p_values)))
  expect_null(po_single$state$p_value_scope)

  dropped = mlr3::TaskUnsupervised$new("mbspca_dropped", backend = data.frame(x, z1 = 1, z2 = 1))
  po_dropped = PipeOpMBsPCA$new(
    blocks = list(a = colnames(x), z = c("z1", "z2")),
    param_vals = list(ncomp = 2L, permutation_test = TRUE, n_perm = 19L)
  )
  expect_warning(po_dropped$train(list(dropped)), "at least two usable blocks")
  expect_identical(po_dropped$state$ncomp, 2L)
  expect_true(all(is.na(po_dropped$state$p_values)))
})


test_that("PipeOpMBsPCA centres blocks and names weights by block", {
  blocks = list(eng = c("disp", "hp", "drat"), body = c("wt", "qsec"))
  data = mtcars[, unlist(blocks)]
  task = mlr3::TaskUnsupervised$new("mbspca_mtcars", backend = data)
  po_raw = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2L))
  po_raw$train(list(task))

  centred = mlr3::TaskUnsupervised$new(
    "mbspca_mtcars_centred",
    backend = as.data.frame(scale(data, scale = FALSE))
  )
  po_centred = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2L))
  po_centred$train(list(centred))

  expect_equal(po_raw$state$weights, po_centred$state$weights, tolerance = 1e-8)
  expect_equal(po_raw$state$ev_block, po_centred$state$ev_block, tolerance = 1e-8)
  expect_equal(po_raw$state$center$body, colMeans(data[, blocks$body]))
  # The body weights are not the column-mean direction of the raw data.
  w_body = po_raw$state$weights$PC1[["body"]]
  mu_body = colMeans(data[, blocks$body])
  expect_lt(abs(sum(w_body * mu_body)) / sqrt(sum(mu_body^2)), 0.9)
  expect_named(po_raw$state$weights$PC1, names(blocks))
  expect_named(po_raw$state$converged, c("PC1", "PC2"))
})


test_that("PipeOpMBsPCA caps c_matrix budgets at the retained width", {
  set.seed(52)
  n = 40L
  data = data.frame(
    a1 = rnorm(n), a2 = rnorm(n), a3 = rnorm(n), a4 = 1,
    b1 = rnorm(n), b2 = rnorm(n)
  )
  task = mlr3::TaskUnsupervised$new("mbspca_capped", backend = data)
  cm = matrix(c(2, 1.2), ncol = 1L, dimnames = list(c("a", "b"), "PC1"))
  pipeop = PipeOpMBsPCA$new(
    blocks = list(a = c("a1", "a2", "a3", "a4"), b = c("b1", "b2")),
    param_vals = list(c_matrix = cm)
  )
  expect_no_error(pipeop$train(list(task)))
  expect_equal(unname(pipeop$state$c_matrix["a", 1L]), sqrt(3))
})


test_that("PipeOpMBsPCA warns when the solver does not converge", {
  task = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 106L)
  pipeop = PipeOpMBsPCA$new(
    blocks = task$block_features(),
    param_vals = list(ncomp = 1L, max_iter = 1L)
  )
  expect_warning(pipeop$train(list(task)), "did not converge within 1 iterations for component\\(s\\) PC1")
  expect_false(pipeop$state$converged[["PC1"]])
})
