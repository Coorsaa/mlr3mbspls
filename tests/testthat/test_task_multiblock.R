test_that("packaged synthetic multiblock tasks are registered", {
  expect_true(all(c(
    "mbspls_synthetic_blocks",
    "mbspls_synthetic_classif",
    "mbspls_synthetic_regr"
  ) %in% mlr3::mlr_tasks$keys()))
})


test_that("task_multiblock_synthetic creates unsupervised and supervised tasks", {
  task_clust = task_multiblock_synthetic(task_type = "clust", n = 30L, seed = 1L)
  expect_true(inherits(task_clust, "TaskClust"))
  expect_setequal(task_clust$block_names, c("block_a", "block_b", "block_c"))
  expect_setequal(names(task_clust$block_features()), c("block_a", "block_b", "block_c"))

  mats = task_clust$block_data(as_matrix = TRUE)
  expect_true(is.list(mats))
  expect_equal(length(mats), 3L)
  expect_true(all(vapply(mats, is.matrix, logical(1))))
  expect_true(all(vapply(mats, nrow, integer(1)) == task_clust$nrow))

  task_classif = task_multiblock_synthetic(task_type = "classif", n = 30L, seed = 1L)
  expect_true(inherits(task_classif, "TaskClassif"))
  expect_equal(task_classif$target_names, "subtype")
  expect_false("subtype" %in% task_classif$feature_names)
  expect_setequal(task_classif$class_names, c("Basal", "Her2", "LumA"))
  expect_setequal(task_classif$block_names, c("block_a", "block_b", "block_c"))

  task_regr = task_multiblock_synthetic(task_type = "regr", n = 30L, seed = 1L)
  expect_true(inherits(task_regr, "TaskRegr"))
  expect_equal(task_regr$target_names, "response")
  expect_false("response" %in% task_regr$feature_names)
  expect_setequal(task_regr$block_names, c("block_a", "block_b", "block_c"))
})


test_that("TaskMultiBlock factory flattens list-of-block input", {
  set.seed(1)
  x = list(
    clin = data.frame(age = rnorm(20), bmi = rnorm(20)),
    omics = data.frame(g1 = rnorm(20), g2 = rnorm(20), g3 = rnorm(20)),
    prot = data.frame(p1 = rnorm(20), p2 = rnorm(20))
  )
  y = factor(sample(c("A", "B"), 20L, replace = TRUE))

  task = TaskMultiBlock(x, target = y, task_type = "classif", id = "toy_mb")

  expect_true(inherits(task, "TaskClassif"))
  expect_equal(task$id, "toy_mb")
  expect_setequal(task$block_names, c("clin", "omics", "prot"))
  expect_true(all(grepl("^clin__", task$block_features("clin"))))
  expect_true(all(grepl("^omics__", task$block_features("omics"))))
  expect_true(all(grepl("^prot__", task$block_features("prot"))))
})


test_that("task_multiblock_synthetic preserves the caller RNG state", {
  set.seed(42)
  seed_before = get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  invisible(task_multiblock_synthetic(task_type = "clust", n = 24L, seed = 7L))
  seed_after = get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  expect_equal(seed_after, seed_before)
})


test_that("TaskMultiBlock validates list-of-block row alignment", {
  x_bad = list(
    a = data.frame(x = rnorm(5)),
    b = data.frame(y = rnorm(4))
  )

  expect_error(
    TaskMultiBlock(x_bad, task_type = "clust"),
    "same number of rows"
  )
})


test_that("TaskMultiBlock auto-infers task type and preserves classif positive class from source task", {
  dt = data.table::data.table(
    f1 = rnorm(20),
    f2 = rnorm(20),
    g1 = rnorm(20),
    y = factor(sample(c("neg", "pos"), 20, TRUE), levels = c("neg", "pos"))
  )
  blocks = list(a = c("f1", "f2"), b = c("g1"))

  task0 = TaskMultiBlock(dt, blocks = blocks, target = "y", task_type = "classif", id = "src")
  task0$positive = "pos"

  task1 = TaskMultiBlock(task0, task_type = "auto", id = "copy")
  expect_true(inherits(task1, "TaskClassif"))
  expect_equal(task1$target_names, "y")
  expect_equal(task1$positive, "pos")
  expect_setequal(task1$block_names, names(blocks))
})


test_that("TaskMultiBlock supports auto regression inference and block_data subset by block names", {
  dt = data.table::data.table(
    a1 = rnorm(30),
    a2 = rnorm(30),
    b1 = rnorm(30),
    b2 = rnorm(30)
  )
  blocks = list(a = c("a1", "a2"), b = c("b1", "b2"))
  y = rnorm(30)

  task = TaskMultiBlock(dt, blocks = blocks, target = y, task_type = "auto", target_name = "y")
  expect_true(inherits(task, "TaskRegr"))
  expect_equal(task$target_names, "y")

  sub = task$block_data(blocks = c("b"), as_matrix = TRUE)
  expect_named(sub, "b")
  expect_true(is.matrix(sub$b))
  expect_equal(colnames(sub$b), blocks$b)
})
