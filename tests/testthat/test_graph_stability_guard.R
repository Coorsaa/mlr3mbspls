test_that("mbspls_graph errors when stability_only=TRUE and bootstrap_selection=FALSE", {
  expect_error(
    mbspls_graph(
      blocks = list(b1 = letters[1:4], b2 = letters[5:8]),
      ncomp = 1L,
      stability_only = TRUE,
      bootstrap_selection = FALSE
    ),
    "stability_only"
  )
})


test_that("mbspls_graph_learner errors when stability_only=TRUE and bootstrap_selection=FALSE", {
  expect_error(
    mbspls_graph_learner(
      blocks = list(b1 = letters[1:4], b2 = letters[5:8]),
      ncomp = 1L,
      stability_only = TRUE,
      bootstrap_selection = FALSE
    ),
    "stability_only"
  )
})


test_that("mbspls_graph succeeds when stability_only=FALSE and bootstrap_selection=FALSE", {
  g = mbspls_graph(
    blocks = list(b1 = letters[1:4], b2 = letters[5:8]),
    ncomp = 1L,
    stability_only = FALSE,
    bootstrap_selection = FALSE
  )
  expect_true(inherits(g, "Graph"))
})


test_that("mbspls_graph succeeds when stability_only=TRUE and bootstrap_selection=TRUE", {
  g = mbspls_graph(
    blocks = list(b1 = letters[1:4], b2 = letters[5:8]),
    ncomp = 1L,
    stability_only = TRUE,
    bootstrap_selection = TRUE
  )
  expect_true(inherits(g, "Graph"))
})

test_that("mbspls_graph exposes safe training-block retention", {
  blocks = list(b1 = letters[1:4], b2 = letters[5:8])
  graph = mbspls_graph(
    blocks = blocks,
    ncomp = 1L,
    bootstrap = FALSE,
    store_train_blocks = TRUE,
    bootstrap_selection = FALSE
  )

  expect_true(
    graph$pipeops$mbspls$param_set$values$store_train_blocks
  )
  expect_error(
    mbspls_graph(
      blocks = blocks,
      ncomp = 1L,
      bootstrap = TRUE,
      store_train_blocks = FALSE,
      bootstrap_selection = TRUE
    ),
    "must be TRUE"
  )
})

test_that("graph constructors and the bootstrap PipeOp agree on align levels and seeding", {
  levels_po = po("mbspls_bootstrap_select")$param_set$levels$align
  expect_setequal(eval(formals(mbspls_graph)$align), levels_po)
  expect_setequal(eval(formals(mbspls_graph_learner)$align), levels_po)

  effective_seed = function(ps) {
    utils::modifyList(
      paradox::default_values(ps),
      ps$get_values(tags = "train"),
      keep.null = TRUE
    )$seed_bootstrap
  }
  graph = mbspls_graph(blocks = list(b1 = letters[1:4], b2 = letters[5:8]), ncomp = 1L)
  expect_null(effective_seed(po("mbspls_bootstrap_select")$param_set))
  expect_null(effective_seed(graph$pipeops$mbspls_bootstrap_select$param_set))
})
