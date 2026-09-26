test_that("block columns expand encoded base names by literal prefix", {
  resolve = mlr3mbspls:::mb_resolve_block_columns

  got = resolve(
    c("site.A", "site.B", "siteX", "a.b.c", "a.bx", "axb.1"),
    list(s = "site", a = "a.b")
  )
  expect_identical(got, list(s = c("site.A", "site.B"), a = "a.b.c"))

  # Regular-expression metacharacters are matched literally.
  got = resolve(
    c("x(1).lvl", "x11.lvl", "x+y.a", "xxy.a", "x1"),
    list(b = c("x(1)", "x+y", "x1"))
  )
  expect_identical(got$b, c("x(1).lvl", "x+y.a", "x1"))

  # A declared name present in the data maps exactly to itself.
  expect_identical(resolve(c("x", "x.1"), list(b = "x"))$b, "x")

  # Expanded columns keep the data order at the position of their base name.
  expect_identical(
    resolve(c("g.z", "a1", "g.a", "a2"), list(b = c("a1", "g", "a2")))$b,
    c("a1", "g.z", "g.a", "a2")
  )

  # Blocks without matches are returned empty rather than dropped.
  expect_identical(resolve(c("a1", "b1"), list(a = "a1", b = "zz")), list(a = "a1", b = character(0)))
})


test_that("prefix expansion never claims columns declared in another block", {
  resolve = mlr3mbspls:::mb_resolve_block_columns

  # Encoded factor `sex` next to a declared numeric `sex.hormone`.
  got = resolve(
    c("sex.m", "sex.x", "sex.hormone", "a1", "b1"),
    list(A = c("sex", "a1"), B = c("sex.hormone", "b1"))
  )
  expect_identical(got, list(A = c("sex.m", "sex.x", "a1"), B = c("sex.hormone", "b1")))

  # A feature dropped upstream must not pull in another block's feature.
  got = resolve(
    c("age.onset", "a1", "a2", "b1", "b2"),
    list(A = c("age", "a1", "a2"), B = c("age.onset", "b1", "b2"))
  )
  expect_identical(got, list(A = c("a1", "a2"), B = c("age.onset", "b1", "b2")))

  # A column matching several absent base names belongs to the longest one.
  got = resolve(
    c("sex.m", "sex.hormone.high", "a1"),
    list(A = c("sex", "a1"), B = "sex.hormone")
  )
  expect_identical(got, list(A = c("sex.m", "a1"), B = "sex.hormone.high"))
})


test_that("resolved blocks must be disjoint", {
  resolve = mlr3mbspls:::mb_resolve_block_columns
  expect_error(
    resolve(c("x.1", "x.2", "y"), list(A = "x", B = c("x", "y"))),
    "disjoint across blocks.*x\\.1 \\(A, B\\)"
  )
  expect_error(
    resolve(c("x", "y"), list(A = "x", B = c("x", "y"))),
    "disjoint across blocks.*x \\(A, B\\)"
  )
})


test_that("mb_resolve_blocks resolves encoded columns across the whole mapping", {
  dt = data.table::data.table(
    sex.m = c(0, 1, 0, 1),
    sex.x = c(1, 0, 0, 0),
    sex.hormone = c(1, 2, 3, 5),
    a1 = c(2, 1, 4, 3),
    b1 = c(1, 3, 2, 4)
  )
  got = mlr3mbspls:::mb_resolve_blocks(
    dt,
    list(A = c("sex", "a1"), B = c("sex.hormone", "b1"))
  )
  expect_identical(got, list(A = c("sex.m", "sex.x", "a1"), B = c("sex.hormone", "b1")))
})


make_encoded_site_task = function() {
  set.seed(7)
  n = 60L
  site = factor(rep(c("A", "B"), each = n / 2L))
  dt = data.table::data.table(
    site = site,
    clin_age = stats::rnorm(n) + (site == "B"),
    clin_sex = factor(ifelse(stats::runif(n) < ifelse(site == "B", 0.8, 0.2), "M", "F")),
    clin_smoke = factor(sample(c("no", "yes", "ex"), n, replace = TRUE)),
    omics_1 = stats::rnorm(n),
    omics_2 = stats::rnorm(n),
    omics_3 = stats::rnorm(n)
  )
  blocks = list(
    clin = c("clin_age", "clin_sex", "clin_smoke"),
    omics = c("omics_1", "omics_2", "omics_3")
  )
  task = TaskMultiBlock(dt, blocks = blocks, task_type = "clust")
  encode = mlr3pipelines::po("encode",
    method = "treatment",
    affect_columns = mlr3pipelines::selector_invert(mlr3pipelines::selector_name("site"))
  )
  list(task = encode$train(list(task))[[1L]], blocks = blocks, site = site)
}


test_that("TaskMultiBlock materialises encoded factor columns within their block", {
  setup = make_encoded_site_task()
  task = setup$task
  clin_encoded = c("clin_age", "clin_sex.M", "clin_smoke.no", "clin_smoke.yes")

  expect_identical(task$blocks, setup$blocks)
  expect_identical(task$block_features(materialize = TRUE)$clin, clin_encoded)
  expect_identical(task$block_features("clin", materialize = TRUE), clin_encoded)
  expect_identical(names(task$block_data()$clin), clin_encoded)
  expect_identical(colnames(task$block_data(blocks = "clin", as_matrix = TRUE)$clin), clin_encoded)

  qc = task$overview()
  expect_identical(qc$blocks$n_features[qc$blocks$block == "clin"], 4L)
  expect_identical(qc$overview$n_features, 7L)
  qc_subset = mb_task_overview(task, blocks = "clin")
  expect_identical(qc_subset$blocks$n_features, 4L)
})


test_that("site correction and block scaling include encoded factor columns", {
  setup = make_encoded_site_task()
  task = setup$task
  clin_encoded = c("clin_age", "clin_sex.M", "clin_smoke.no", "clin_smoke.yes")
  site_b = as.numeric(setup$site == "B")

  # The fixture makes the sex dummy strongly site-confounded.
  expect_gt(abs(stats::cor(task$data()$clin_sex.M, site_b)), 0.3)

  sitecorr = mlr3pipelines::po("sitecorr",
    blocks = setup$blocks,
    site_correction = list(clin = "site"),
    method = list(clin = "partial_corr")
  )
  corrected = sitecorr$train(list(task))[[1L]]
  expect_identical(sitecorr$state$blocks$clin, clin_encoded)
  expect_lt(abs(stats::cor(corrected$data()$clin_sex.M, site_b)), 1e-8)

  blockscale = mlr3pipelines::po("blockscale", blocks = setup$blocks, method = "unit_ssq")
  scaled = blockscale$train(list(task))[[1L]]
  expect_identical(blockscale$state$blocks$clin, clin_encoded)
  x_clin = as.matrix(scaled$data(cols = clin_encoded))
  expect_equal(sum(x_clin^2), 1, tolerance = 1e-10)
})
