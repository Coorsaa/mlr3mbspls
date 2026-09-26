test_that("sequential tuning EV uses the original held-out variance denominator", {
  z = as.numeric(scale(seq_len(20L)))
  u = as.numeric(residuals(lm(rep(c(-1, 1), 10L) ~ z)))
  u = u * sqrt(sum(z^2) / sum(u^2))
  X = list(a = cbind(strong = 10 * z, weak = u), b = cbind(strong = 10 * z, weak = u))
  W = list(list(c(1, 0), c(1, 0)), list(c(0, 1), c(0, 1)))
  original_ss = vapply(X, function(block) sum(block^2), numeric(1L))

  for (constructor in list(TunerSeqMBsPLS, TunerSeqMBsPCA)) {
    tuner = constructor$new(early_stopping = FALSE)
    private = tuner$.__enclos_env__$private
    residual = private$.deflate_blocks(X, W[[1L]])
    args = list(X_train_fit = residual, X_test = residual,
      W_list = W[[2L]], original_ss = original_ss)
    if (inherits(tuner, "TunerSeqMBsPLS")) args$correlation_method = "pearson"
    payload = do.call(private$.one_lv_payload, args)

    P = list(private$.compute_train_loadings(X, W[[1L]]),
      private$.compute_train_loadings(residual, W[[2L]]))
    full = compute_test_ev(X, W, P, loading_source = "train")
    expect_equal(as.numeric(payload$ev_comp), 1 / 101, tolerance = 1e-10)
    expect_equal(as.numeric(payload$ev_block), as.numeric(full$ev_block[2L, ]))
    expect_equal(as.numeric(payload$ev_comp), as.numeric(full$ev_comp[2L]))
  }
})

test_that("sequential tuners derive feature schemas inside each training fold", {
  set.seed(701L)
  data = as.data.frame(matrix(rnorm(24L * 6L), 24L,
    dimnames = list(NULL, c("x1", "x2", "x3", "z1", "z2", "z3"))))
  task = mlr3cluster::TaskClust$new("fold_schema", data)
  blocks = list(x = c("x1", "x2", "x3"), z = c("z1", "z2", "z3"))
  # Simulate a learned filter whose selected columns differ across fit samples.
  selector = function(task) {
    if (task$nrow > 12L) c("x1", "x2", "z1", "z2") else c("x2", "z2")
  }
  for (method in c("pls", "pca")) {
    component = if (method == "pls") {
      PipeOpMBsPLS$new(blocks = blocks)
    } else {
      PipeOpMBsPCA$new(blocks = blocks, param_vals = list(max_iter = 7L, tol = 2e-7))
    }
    learner = mlr3::as_learner(
      mlr3pipelines::po("select", selector = selector) %>>% component %>>%
        mlr3pipelines::po("learner", mlr3::lrn("clust.kmeans", centers = 2L))
    )
    instance = mlr3tuning::ti(
      task = task, learner = learner, resampling = mlr3::rsmp("holdout"),
      measure = mlr3::msr(if (method == "pls") "mbspls.mac" else "mbspca.mean_ev"),
      terminator = bbotk::trm("evals", n_evals = 1L)
    )
    constructor = if (method == "pls") TunerSeqMBsPLS else TunerSeqMBsPCA
    expect_no_error(constructor$new(budget = 1L,
      resampling = mlr3::rsmp("cv", folds = 2L),
      early_stopping = FALSE)$optimize(instance))
    # Fold training has one retained column per block, despite wider full data.
    expect_equal(as.numeric(instance$result_learner_param_vals$c_matrix), c(1, 1))
    expect_true(is.finite(instance$result_y))
  }
})
