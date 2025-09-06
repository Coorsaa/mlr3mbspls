#' Default MB‑sPLS Graph
#' @description
#' This function constructs a default graph for MB-sPLS analysis, which includes
#' a site correction step, followed by the MB-sPLS PipeOp, and finally a clustering
#' learner. The graph is designed to handle multi-block data, where each block
#' represents a different set of features.
#' 
#' @param blocks Named list of character vectors, each containing the names of
#'   the features to be used in each block.
#' @param site_correction Named list of character vectors, each containing the
#'   names of the features to be used for site correction.
#' @param site_correction_methods Named list of character vectors, each containing the
#'   names of the methods to be used for site correction.
#'
#' @return [`GraphLearner`]
#' @import mlr3
#' @import mlr3cluster
#' @import mlr3pipelines
#' @import checkmate
#' @export
mbspls_preproc_graph = function(
  blocks,
  site_correction,
  site_correction_methods,
  keep_site_col = FALSE,
  k = 5
) {
  assert_list(blocks, types = "character", names = "unique")
  assert_list(site_correction, types = "character", names = "unique")
  assert_list(site_correction_methods, types = "character", names = "unique")

  graph <- ppl("convert_types", "character", "factor") %>>% 
    po("encode",
       method = "treatment",
       affect_columns = selector_invert(
         selector_name(
           unlist(site_correction, use.names = FALSE)
         )
       )
    ) %>>%
    ppl("imputeknn", k = k) %>>%
    po("sitecorr",
      blocks = blocks,
      site_correction = site_correction,
      method = site_correction_methods,
      keep_site_col = keep_site_col
    ) %>>%
    po("scale")
  graph
}


#' Create a MB‑sPLS GraphLearner
#' @description
#' This function creates a `GraphLearner` for MB-sPLS analysis, which includes
#' preprocessing steps, the MB-sPLS PipeOp, and a downstream learner.
#' 
#' @param blocks Named list of character vectors, each containing the names of
#'   the features to be used in each block.
#' @param site_correction Named list of character vectors, each containing the
#'   names of the features to be used for site correction.
#' @param site_correction_methods Named list of character vectors, each containing the
#'   names of the methods to be used for site correction.
#' @param keep_site_col Logical, whether to keep site correction columns in the output.
#' @param ncomp Integer, number of components to extract in MB-sPLS.
#' @param k Integer, number of neighbors to use for k-NN imputation (default is 5).
#' @param performance_metric Character, performance metric to optimize ("mac" or "frobenius").
#' @param learner An `mlr3` learner to apply after MB-sPLS (default is k-means with 1 center).
#' @param permutation_test Logical, whether to perform permutation testing for component significance.
#' @param n_perm Integer, number of permutations for permutation test (default is 500).
#' @param perm_alpha Numeric, significance level for permutation test (default is 0.05).
#' @param bootstrap_test Logical, whether to perform bootstrap testing.
#' @param boot_alpha Numeric, significance level for bootstrap test (default is 0.05).
#' @param boot_keep_draws Logical, whether to keep bootstrap draws.
#' @param boot_store_vectors Logical, whether to store bootstrap vectors.
#' @param boot_min_selectfreq Numeric, minimum selection frequency for bootstrap.
#' @param c_matrix Matrix, constraint matrix for MB-sPLS.
#' @param n_boot Integer, number of bootstrap samples (default is 500).
#' @param val_test Character, validation test method.
#' @param val_test_alpha Numeric, significance level for validation test (default is 0.05).
#' @param val_test_n Integer, number of validation test iterations (default is 1000).
#' @param val_test_permute_all Logical, whether to permute all variables in validation test.
#' @param ref_block Character, name of the reference block for weight flipping.
#' @param towards Character, direction for weight flipping ("positive" or "negative").
#' @param additional_data Optional additional data to be used in the MBsPLS PipeOp.
#' @param source Character, source for weight flipping ("weights" or "scores").
#' @param log_env Environment for logging (default is NULL).
#' @return [`GraphLearner`]
#' @import mlr3
#' @import mlr3cluster
#' @import mlr3pipelines
#' @import checkmate
#' @export
mbspls_graph_learner = function(
  blocks,
  site_correction,
  site_correction_methods,
  keep_site_col = FALSE,
  ncomp,
  k = 5,
  performance_metric = c("mac", "frobenius"),
  learner = lrn("clust.kmeans", centers = 1L),
  permutation_test = FALSE,
  n_perm = 500,
  perm_alpha = 0.05,
  bootstrap_test = FALSE,
  boot_alpha = 0.05,
  boot_keep_draws = TRUE,
  boot_store_vectors = FALSE,
  boot_min_selectfreq = 0,
  c_matrix = NULL,
  n_boot = 500L,
  val_test = "none",
  val_test_alpha = 0.05,
  val_test_n = 1000,
  val_test_permute_all = TRUE,
  ref_block = names(blocks)[1],
  towards = "positive",
  additional_data = NULL,
  source = "weights",
  log_env = NULL
) {
  assert_list(blocks, types = "character", names = "unique")
  assert_list(site_correction, types = "character", names = "unique")
  assert_list(site_correction_methods, types = "character", names = "unique")
  assert_integerish(ncomp, lower = 1, len = 1)
  assert_integerish(k, lower = 1, len = 1)
  performance_metric <- match.arg(performance_metric)
  assert_class(learner, "Learner")
  assert_logical(permutation_test, len = 1)

  log_env <- if (is.null(log_env)) new.env(parent = emptyenv()) else log_env
  graph <- ppl("mbspls_preproc",
    blocks = blocks,
    site_correction = site_correction,
    site_correction_methods = site_correction_methods,
    keep_site_col = keep_site_col,
    k = k
  ) %>>%
    po("mbspls",
      blocks = blocks,
      ncomp = ncomp,
      performance_metric = performance_metric,
      permutation_test = permutation_test,
      n_perm = n_perm,
      perm_alpha = perm_alpha,
      bootstrap_test = bootstrap_test,
      boot_alpha = boot_alpha,
      boot_keep_draws = boot_keep_draws,
      boot_store_vectors = boot_store_vectors,
      boot_min_selectfreq = boot_min_selectfreq,
      c_matrix = c_matrix,
      n_boot = n_boot,
      val_test = val_test,
      val_test_alpha = val_test_alpha,
      val_test_n = val_test_n,
      val_test_permute_all = val_test_permute_all,
      additional_data = additional_data,
      log_env = log_env
    ) %>>%
    po("mbspls_flipweights",
      ref_block = ref_block,
      towards = towards,
      source = source,
      log_env = log_env
    ) %>>%
    po("learner", learner)
  gl <- GraphLearner$new(graph)
  attr(gl, "log_env") <- log_env
  gl
}


#' Create a k-NN Imputation Graph
#' @description
#' This function creates a `Graph` for k-NN imputation of missing values in
#' both numeric and factor columns. It uses separate k-NN learners for numeric
#' and factor data types to ensure appropriate imputation.
#' @param k Integer, number of neighbors to use for k-NN imputation (default is 5).
#' @return [`Graph`]
#' @import mlr3
#' @import mlr3pipelines
#' @import checkmate
#' @export
impute_knn_graph <- function(k = 5) {
  assert_integerish(k, lower = 1, len = 1)
  
  imp_num = po("imputelearner",
    learner = lrn("regr.knngower", k = k),
    affect_columns = selector_union(
      selector_type("numeric"),
      selector_type("integer")
    )
  )
  imp_num$id = "impute_num_knn"
  imp_fac = po("imputelearner",
    learner = lrn("classif.knngower", k = k),
    affect_columns = selector_type("factor")
  )
  imp_fac$id = "impute_fac_knn"
  imp_num %>>% imp_fac
}

