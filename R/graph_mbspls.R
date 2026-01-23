#' Create a k-NN Imputation Graph
#' @description k-NN imputation for numeric and factor columns.
#' @param k Integer, number of neighbors (default 5).
#' @return [`Graph`]
#' @import mlr3 mlr3pipelines checkmate
#' @export
impute_knn_graph = function(k = 5) {
  checkmate::assert_integerish(k, lower = 1, len = 1)
  imp_num = po("imputelearner",
    learner = lrn("regr.knngower", k = k),
    affect_columns = selector_union(selector_type("numeric"), selector_type("integer"))
  )
  imp_num$id = "impute_num_knn"
  imp_fac = po("imputelearner",
    learner = lrn("classif.knngower", k = k),
    affect_columns = selector_type("factor")
  )
  imp_fac$id = "impute_fac_knn"
  imp_num %>>% imp_fac
}


#' Default MB-sPLS Preprocessing Graph
#' @description
#' Site correction -> encoding/imputation -> scaling. No MB-sPLS here.
#' @param blocks Named list of character vectors per block.
#' @param site_correction Named list of features used for site correction.
#' @param site_correction_methods Named list of methods for site correction.
#' @param keep_site_col Keep site column after correction?
#' @param k k for kNN imputation (default 5).
#' @param id_suffix Optional suffix for PipeOp ids.
#' @return [`Graph`]
#' @import mlr3 mlr3pipelines checkmate
#' @export
mbspls_preproc_graph = function(
  blocks,
  site_correction,
  site_correction_methods,
  keep_site_col = FALSE,
  k = 5,
  id_suffix = NULL
) {
  assert_list(blocks, types = "character", names = "unique")
  assert_list(site_correction, types = c("character", "list"), names = "unique")
  assert_list(site_correction_methods, types = "character", names = "unique")

  ppl_convert_types = ppl("convert_types", "character", "factor")
  ppl_impute = ppl("imputeknn", k = k)

  if (!is.null(id_suffix)) {
    ppl_convert_types$update_ids(postfix = paste0("_", id_suffix))
    ppl_impute$update_ids(postfix = paste0("_", id_suffix))
  }

  graph = ppl_convert_types %>>%
    po("encode",
      id = paste0("encode_", id_suffix),
      method = "treatment",
      affect_columns = selector_invert(selector_name(unlist(site_correction, use.names = FALSE)))
    ) %>>%
    ppl_impute %>>%
    po("sitecorr",
      id = paste0("sitecorr_", id_suffix),
      blocks = blocks,
      site_correction = site_correction,
      method = site_correction_methods,
      keep_site_col = keep_site_col
    ) %>>%
    po("scale",
      id = paste0("scale_", id_suffix)
    )
  graph
}


#' MB-sPLS Graph: preproc -> MB-sPLS -> bootstrap-select (two selection methods)
#'
#' @param blocks,site_correction,site_correction_methods,keep_site_col Preproc settings.
#' @param ncomp Number of MB-sPLS components.
#' @param k k-NN for imputation (preproc).
#' @param performance_metric "mac" or "frobenius".
#' @param correlation_method "pearson" or "spearman".
#' @param c_matrix Optional L1 constraint matrix for MB-sPLS.
#'
#' @param permutation_test,n_perm,perm_alpha Train-time permutation test (MB-sPLS).
#' @param predict_weights character; one of "auto","raw","stable_ci","stable_frequency".
#'   Controls which weights PipeOpMBsPLS uses at predict/validation time.
#' @param val_test,val_test_alpha,val_test_n,val_test_permute_all Prediction-side validation (MB-sPLS).
#'
#' @param bootstrap Logical; run bootstrap selection (default TRUE).
#' @param bootstrap_selection Logical; whether to run bootstrap-based feature
#'   selection inside [PipeOpMBsPLSBootstrapSelect] (default TRUE).
#' @param stability_only Logical; only compute stability, no selection (default FALSE).
#' @param B Integer; bootstrap replicates (default 500).
#' @param alpha Numeric; CI alpha (default 0.05).
#' @param align "block_sign" (default) or "global_correlation".
#' @param selection_method "ci" (default) or "frequency".
#' @param frequency_threshold Numeric in `[0,1]`; only if selection_method="frequency" (default 0.6).
#' @param stable_weight_source "training" (default) or "bootstrap_mean".
#' @param stratify_by_block Optional dummy block for stratified bootstrap (e.g., "Studygroup").
#' @param workers Integer; Unix workers (default cores-1).
#' @param seed_train Optional seed for MB-sPLS training.
#' @param seed_bootstrap Optional seed for bootstrap selection.
#' @param id_suffix Optional suffix for PipeOp ids.
#' @param log_env Shared environment (created if NULL).
#'
#' @return [mlr3pipelines::Graph]
#' @import mlr3 mlr3pipelines checkmate
#' @export
mbspls_graph = function(
  blocks,
  site_correction,
  site_correction_methods,
  keep_site_col = FALSE,
  ncomp,
  k = 5,
  performance_metric = c("mac", "frobenius"),
  correlation_method = c("pearson", "spearman"),
  c_matrix = NULL,

  permutation_test = FALSE,
  n_perm = 500L,
  perm_alpha = 0.05,
  predict_weights = c("auto", "raw", "stable_ci", "stable_frequency"),
  val_test = c("none", "permutation", "bootstrap"),
  val_test_alpha = 0.05,
  val_test_n = 1000L,
  val_test_permute_all = TRUE,

  bootstrap = TRUE,
  stability_only = FALSE,
  B = 500L,
  alpha = 0.05,
  align = c("block_sign", "score_correlation"),
  bootstrap_selection = TRUE,
  selection_method = c("ci", "frequency"),
  frequency_threshold = 0.60,
  stable_weight_source = c("training", "bootstrap_mean"),
  stratify_by_block = NULL,
  seed_train = NULL,
  seed_bootstrap = NULL,
  workers = 1L,
  id_suffix = NULL,
  log_env = NULL
) {
  checkmate::assert_list(blocks, types = "character", names = "unique")
  checkmate::assert_list(site_correction, types = c("character", "list"), names = "unique")
  checkmate::assert_list(site_correction_methods, types = "character", names = "unique")
  checkmate::assert_int(ncomp, lower = 1)
  performance_metric = match.arg(performance_metric)
  correlation_method = match.arg(correlation_method)
  predict_weights = match.arg(predict_weights)
  if (isTRUE(stability_only) && identical(predict_weights, "auto")) {
    predict_weights = "raw"
  }
  val_test = match.arg(val_test)
  align = match.arg(align)
  selection_method = match.arg(selection_method)
  stable_weight_source = match.arg(stable_weight_source)

  log_env = if (is.null(log_env)) new.env(parent = emptyenv()) else log_env

  if (isTRUE(bootstrap_selection)) {
    po_bootstrap_select = po("mbspls_bootstrap_select",
      id = if (is.null(id_suffix)) "mbspls_bootstrap_select" else paste0("mbspls_bootstrap_select_", id_suffix),
      log_env = log_env,
      bootstrap = bootstrap,
      stability_only = stability_only,
      B = B,
      alpha = alpha,
      align = align,
      selection_method = selection_method,
      frequency_threshold = frequency_threshold,
      stable_weight_source = stable_weight_source,
      stratify_by_block = stratify_by_block,
      seed_bootstrap = seed_bootstrap,
      workers = workers
    )
  }

  graph = ppl("mbspls_preproc",
    blocks = blocks,
    site_correction = site_correction,
    site_correction_methods = site_correction_methods,
    keep_site_col = keep_site_col,
    k = k,
    id_suffix = id_suffix
  ) %>>%
    po("mbspls",
      id = if (is.null(id_suffix)) "mbspls" else paste0("mbspls_", id_suffix),
      blocks = blocks,
      ncomp = ncomp,
      performance_metric = performance_metric,
      correlation_method = correlation_method,
      c_matrix = c_matrix,

      # optional train-time permutation
      permutation_test = permutation_test,
      n_perm = n_perm,
      perm_alpha = perm_alpha,

      # which weights to use at predict/validation time?
      predict_weights = predict_weights,

      # optional prediction-side validation
      val_test = val_test,
      val_test_alpha = val_test_alpha,
      val_test_n = val_test_n,
      val_test_permute_all = val_test_permute_all,

      # expose training snapshot for selection
      store_train_blocks = isTRUE(bootstrap),
      append = isTRUE(bootstrap_selection) && isTRUE(bootstrap) && !isTRUE(stability_only),
      seed_train = seed_train,
      log_env = log_env
    )
  if (isTRUE(bootstrap_selection)) {
    graph = graph %>>% po_bootstrap_select
  }
  return(graph)
}




#' MB-sPLS GraphLearner: preproc -> MB-sPLS -> bootstrap-select (two selection methods) -> learner
#'
#' @param learner Downstream learner (default k-means with 1 center).
#' @inheritParams mbspls_graph
#'
#' @return [mlr3pipelines::GraphLearner]
#' @import mlr3 mlr3cluster mlr3pipelines checkmate
#' @export
mbspls_graph_learner = function(
  learner = lrn("clust.kmeans", centers = 1L),
  blocks,
  site_correction,
  site_correction_methods,
  keep_site_col = FALSE,
  ncomp,
  k = 5,
  performance_metric = c("mac", "frobenius"),
  correlation_method = c("pearson", "spearman"),
  c_matrix = NULL,

  permutation_test = FALSE,
  n_perm = 500L,
  perm_alpha = 0.05,
  predict_weights = c("auto", "raw", "stable_ci", "stable_frequency"),
  val_test = c("none", "permutation", "bootstrap"),
  val_test_alpha = 0.05,
  val_test_n = 1000L,
  val_test_permute_all = TRUE,

  bootstrap = TRUE,
  stability_only = FALSE,
  B = 500L,
  alpha = 0.05,
  align = c("block_sign", "score_correlation"),
  bootstrap_selection = TRUE,
  selection_method = c("ci", "frequency"),
  frequency_threshold = 0.60,
  stable_weight_source = c("training", "bootstrap_mean"),
  stratify_by_block = NULL,
  seed_train = NULL,
  seed_bootstrap = NULL,
  workers = 1L,
  id_suffix = NULL,
  log_env = NULL
) {
  checkmate::assert_class(learner, "Learner")

  graph = mbspls_graph(
    blocks = blocks,
    site_correction = site_correction,
    site_correction_methods = site_correction_methods,
    keep_site_col = keep_site_col,
    ncomp = ncomp,
    k = k,
    performance_metric = performance_metric,
    correlation_method = correlation_method,
    c_matrix = c_matrix,
    permutation_test = permutation_test,
    n_perm = n_perm,
    perm_alpha = perm_alpha,
    predict_weights = predict_weights,
    val_test = val_test,
    val_test_alpha = val_test_alpha,
    val_test_n = val_test_n,
    val_test_permute_all = val_test_permute_all,
    bootstrap = bootstrap,
    stability_only = stability_only,
    B = B,
    alpha = alpha,
    align = align,
    bootstrap_selection = bootstrap_selection,
    selection_method = selection_method,
    frequency_threshold = frequency_threshold,
    stable_weight_source = stable_weight_source,
    stratify_by_block = stratify_by_block,
    seed_train = seed_train,
    seed_bootstrap = seed_bootstrap,
    workers = workers,
    id_suffix = id_suffix,
    log_env = log_env
  )
  as_learner(graph %>>%
    po("learner", learner))
}
