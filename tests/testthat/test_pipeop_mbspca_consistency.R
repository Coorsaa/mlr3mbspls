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

  X_cur = lapply(st$blocks, function(cols) {
    M = as.matrix(task$data(cols = cols))
    storage.mode(M) = "double"
    M
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
    "rows must cover all retained blocks"
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

  cm = matrix(2, nrow = 2L, ncol = 1L)
  po = PipeOpMBsPCA$new(
    blocks = blocks,
    param_vals = list(ncomp = 1L, c_matrix = cm)
  )

  expect_no_error(po$train(list(task)))
  expect_equal(names(po$state$blocks), names(blocks)[1:2])
  expect_equal(po$state$ncomp, 1L)
})
