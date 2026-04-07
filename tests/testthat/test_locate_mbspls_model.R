test_that(".mbspls_pipeop_id requires an explicit id when multiple MB-sPLS nodes exist", {
  graph_like = list(
    pipeops = list(
      m1 = PipeOpMBsPLS$new(blocks = list(b1 = c("x1", "x2"))),
      m2 = PipeOpMBsPLS$new(id = "m2", blocks = list(b1 = c("x1", "x2")))
    )
  )

  expect_error(
    .mbspls_pipeop_id(graph_like, where = "graph_like"),
    "Multiple PipeOpMBsPLS nodes"
  )
  expect_identical(
    .mbspls_pipeop_id(graph_like, mbspls_id = "m2", where = "graph_like"),
    "m2"
  )
})
