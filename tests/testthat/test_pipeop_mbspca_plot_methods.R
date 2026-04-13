test_that("PipeOpMBsPCA plot_score_network errors with informative message when igraph/ggraph absent", {
  skip_if(requireNamespace("igraph", quietly = TRUE) && requireNamespace("ggraph", quietly = TRUE),
    "igraph and ggraph are both installed; cannot test the missing-package guard")

  set.seed(42)
  n = 30
  b1 = matrix(rnorm(n * 3), nrow = n, dimnames = list(NULL, paste0("x", 1:3)))
  b2 = matrix(rnorm(n * 3), nrow = n, dimnames = list(NULL, paste0("z", 1:3)))
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2L))
  df = data.frame(b1, b2)
  task = mlr3::TaskUnsupervised$new(id = "mb_pca", backend = df)
  po$train(list(task))

  if (!requireNamespace("igraph", quietly = TRUE)) {
    expect_error(po$plot_score_network(), "igraph")
  }
  if (!requireNamespace("ggraph", quietly = TRUE)) {
    expect_error(po$plot_score_network(), "ggraph")
  }
})


test_that("PipeOpMBsPCA plot_score_network produces a ggplot when packages are available", {
  skip_if_not_installed("igraph")
  skip_if_not_installed("ggraph")
  skip_if_not_installed("ggplot2")

  set.seed(5)
  n = 40
  # Create correlated blocks so there will be edges in the network
  latent = rnorm(n)
  b1 = cbind(latent + rnorm(n, 0.1), rnorm(n), latent * 0.8 + rnorm(n, 0.1))
  b2 = cbind(latent + rnorm(n, 0.1), rnorm(n), latent * 0.8 + rnorm(n, 0.1))
  colnames(b1) = paste0("x", 1:3)
  colnames(b2) = paste0("z", 1:3)
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2L))
  df = data.frame(b1, b2)
  task = mlr3::TaskUnsupervised$new(id = "mb_pca2", backend = df)
  po$train(list(task))

  g = po$plot_score_network(cutoff = 0.0, method = "pearson")
  expect_s3_class(g, "ggplot")
})


test_that("PipeOpMBsPCA plot_scree and plot_variance return ggplot objects", {
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("scales")

  set.seed(11)
  n = 30
  b1 = matrix(rnorm(n * 4), nrow = n, dimnames = list(NULL, paste0("x", 1:4)))
  b2 = matrix(rnorm(n * 4), nrow = n, dimnames = list(NULL, paste0("z", 1:4)))
  blocks = list(b1 = colnames(b1), b2 = colnames(b2))

  po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 2L))
  df = data.frame(b1, b2)
  task = mlr3::TaskUnsupervised$new(id = "mb_pca3", backend = df)
  po$train(list(task))

  expect_s3_class(po$plot_scree(type = "component"), "ggplot")
  expect_s3_class(po$plot_scree(type = "cumulative"), "ggplot")
  expect_s3_class(po$plot_variance(), "ggplot")
  expect_s3_class(po$plot_score_heatmap(), "ggplot")
})
