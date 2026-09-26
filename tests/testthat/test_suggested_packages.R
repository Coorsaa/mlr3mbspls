test_that("suggested-package guard names missing packages with an install hint", {
  require_suggested = getFromNamespace(".mbspls_require_suggested", "mlr3mbspls")
  testthat::local_mocked_bindings(
    .mbspls_has_namespace = function(pkg) !pkg %in% c("dplyr", "tibble"),
    .package = "mlr3mbspls"
  )

  expect_invisible(require_suggested("stats", "a summary"))
  expect_error(
    require_suggested(c("stats", "dplyr", "tibble"), "mbspls_extract_bootstrap_means()"),
    paste0(
      "mbspls_extract_bootstrap_means\\(\\) requires the suggested package\\(s\\) ",
      "'dplyr', 'tibble'\\. Install with install.packages\\(c\\(\"dplyr\", \"tibble\"\\)\\)\\."
    )
  )
})


test_that("suggested-package guard checks real namespaces", {
  require_suggested = getFromNamespace(".mbspls_require_suggested", "mlr3mbspls")
  expect_true(require_suggested("stats", "a summary"))
  expect_error(
    require_suggested("mlr3mbsplsNotAPackage", "a plot"),
    "a plot requires the suggested package\\(s\\) 'mlr3mbsplsNotAPackage'"
  )
})
