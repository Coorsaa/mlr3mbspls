# Multi-block analysis example with MB-sPLS.
# Run from the package root with:
# Rscript inst/examples/scientific_analysis.R

suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3cluster)
  library(mlr3tuning)
  library(mlr3mbspls)
  library(data.table)
  library(ggplot2)
})

# Keep package logs quiet enough for the main results.
lgr::lgr$set_threshold("warn")
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

# Avoid automatic Rplots.pdf files during non-interactive checks.
if (!interactive()) {
  null_device = if (.Platform$OS.type == "windows") "NUL" else "/dev/null"
  options(device = function(...) grDevices::pdf(file = null_device))
}

# Use fixed seeds for a reproducible demonstration.
set.seed(2026)

# Use small settings so the full analysis runs quickly.
n_components = 2L
n_clusters = 2L
tuning_budget = 2L
bootstrap_replicates = 12L

# Load a synthetic multi-block cohort.
task_train = task_multiblock_synthetic(
  task_type = "clust",
  n = 48L,
  seed = 2026L,
  id = "synthetic_precision_medicine_train"
)

# Inspect the analysis task.
print(task_train)
print(task_train$row_ids[1:6])
print(task_train$feature_names)

# Inspect the block structure.
blocks = task_train$block_features()
print(blocks)
print(lengths(blocks))

# Inspect the first rows of the analytic table.
clinical_table = as.data.table(task_train$data(cols = task_train$feature_names))
print(dim(clinical_table))
print(head(clinical_table))

# Check missingness, constants, block size, and site balance.
task_qc = task_train$overview()
print(task_qc$overview)
print(task_qc$blocks)
print(task_qc$issues)

# Define the site variable used for block-wise adjustment.
site_correction = list(
  block_a = "site_batch",
  block_b = "site_batch",
  block_c = "site_batch"
)

# Use partial correlation adjustment in each block.
site_correction_methods = list(
  block_a = "partial_corr",
  block_b = "partial_corr",
  block_c = "partial_corr"
)

# Build the analysis pipeline: site correction, MB-sPLS, bootstrap stability
# selection of block features, and k-means on the stable latent variables. The
# same configuration is validated, tuned, and fitted below. The MB-sPLS fit is
# deterministic, and the fixed seed_bootstrap gives every bootstrap replicate
# its own RNG stream.
make_learner = function(c_matrix = NULL) {
  mbspls_graph_learner(
    learner = lrn("clust.kmeans", centers = n_clusters),
    task = task_train,
    site_correction = site_correction,
    site_correction_methods = site_correction_methods,
    ncomp = n_components,
    c_matrix = c_matrix,
    performance_metric = "mac",
    permutation_test = FALSE,
    val_test = "none",
    bootstrap = TRUE,
    bootstrap_selection = TRUE,
    selection_method = "ci",
    B = bootstrap_replicates,
    seed_bootstrap = 2027L,
    workers = 1L
  )
}

# Define outer and inner resampling. A single holdout split keeps the example
# fast; use k-fold or repeated outer resampling in an analysis.
rs_outer = rsmp("holdout")
rs_inner = rsmp("holdout")

# Estimate held-out latent correlation and explained variance. Each outer
# fold tunes the sparsity and reruns bootstrap selection on its own training
# rows, so the estimate covers the complete pipeline.
nested_result = mbspls_nested_cv(
  task = task_train,
  graphlearner = make_learner(),
  rs_outer = rs_outer,
  rs_inner = rs_inner,
  ncomp = n_components,
  tuner_budget = tuning_budget,
  tuning_early_stop = FALSE,
  performance_metric = "mac",
  val_test = "none",
  store_payload = FALSE
)

# Inspect the validation results before interpreting the final model.
# measure_test_n_undefined counts components whose held-out latent correlation
# was undefined (fewer than two blocks with non-degenerate scores); they score
# zero.
print(nested_result$results[, .(
  split,
  measure_test,
  mac_lv1_test,
  measure_test_n_undefined
)])
print(nested_result$summary_table)

# Tune the block-wise sparsity matrix on all training samples.
tuner = TunerSeqMBsPLS$new(
  tuner = "random_search",
  budget = tuning_budget,
  resampling = rsmp("holdout"),
  parallel = "none",
  early_stopping = FALSE,
  performance_metric = "mac"
)

# Define the final tuning instance.
tuning_instance = ti(
  task = task_train,
  learner = make_learner(),
  resampling = rsmp("insample"),
  measure = msr("mbspls.mac_evwt"),
  terminator = bbotk::trm("evals", n_evals = 1)
)

# Run the sequential component-wise tuner.
tuner$optimize(tuning_instance)

# Inspect the selected sparsity matrix. Its inner score was maximised on the
# same folds and is optimistic; the nested estimate above is the performance
# estimate.
c_matrix_final = tuning_instance$result$learner_param_vals[[1]]$c_matrix
print(c_matrix_final)
print(tuning_instance$result_y)

# Fit the final model with bootstrap stability selection.
gl_final = make_learner(c_matrix = c_matrix_final)

# Train the final graph learner.
gl_final$train(task_train)

# Predict sample clusters in the training cohort.
pred_train = gl_final$predict(task_train)
print(pred_train)

# Summarise the cluster sizes.
cluster_summary = data.table(cluster = as.character(pred_train$partition))
cluster_summary = cluster_summary[, .N, by = cluster][order(cluster)]
print(cluster_summary)

# Extract model summaries for reporting. The component table reports the
# training objective and explained variance; conditional_p_value and
# p_value_scope are NA because the train-time permutation diagnostic is off.
model_summary = mbspls_model_summary(gl_final)
print(model_summary$overview)
print(model_summary$components)
print(model_summary$blocks)
print(gl_final$model$mbspls$converged)

# Inspect the strongest absolute training weights.
weight_summary = copy(model_summary$weights)
weight_summary[, abs_weight := abs(weight)]
print(weight_summary[order(-abs_weight)][1:12])

# Inspect bootstrap stability: aligned bootstrap means, percentile intervals,
# selection frequencies, and the stable weights that define the LV features
# passed to k-means.
if (!is.null(model_summary$stability)) {
  stability_summary = copy(model_summary$stability)
  print(stability_summary[order(component, block, -freq)][1:12])
}

# Evaluate the fitted graph on an independent synthetic cohort.
task_test = task_multiblock_synthetic(
  task_type = "clust",
  n = 24L,
  seed = 2028L,
  id = "synthetic_precision_medicine_test"
)

# Compute out-of-sample explained variance and latent correlation of the
# stability-selected weights the pipeline uses. A component whose stable
# weights keep fewer than two blocks has no cross-block correlation (NaN).
external_eval = mbspls_eval_new_data(gl_final, task_test)
print(external_eval$weights_source)
print(external_eval$ev_comp)
print(external_eval$ev_block)
print(external_eval$mac_comp)

# Plot raw feature weights.
plot_weights_raw = autoplot(
  gl_final,
  type = "mbspls_weights",
  source = "weights",
  top_n = 6L,
  patch_ncol = 1L
)
print(plot_weights_raw)

# Plot bootstrap-stable feature weights.
plot_weights_stable = autoplot(
  gl_final,
  type = "mbspls_weights",
  source = "bootstrap",
  top_n = 6L,
  patch_ncol = 1L,
  alpha_by_stability = TRUE
)
print(plot_weights_stable)

# Plot bootstrap percentile intervals for feature weights.
plot_weight_ci = mbspls_plot_block_weight_ci(
  gl_final,
  source = "bootstrap",
  top_n = 6L,
  alpha_by_stability = TRUE
)
print(plot_weight_ci)

# Plot explained variance by block and component.
plot_variance = autoplot(
  gl_final,
  type = "mbspls_variance",
  source = "bootstrap",
  show_total = TRUE
)
print(plot_variance)

# Plot cumulative explained variance.
plot_scree = autoplot(
  gl_final,
  type = "mbspls_scree",
  source = "bootstrap",
  cumulative = TRUE
)
print(plot_scree)

# Plot latent correlations across blocks.
plot_heatmap = autoplot(
  gl_final,
  type = "mbspls_heatmap",
  source = "bootstrap",
  method = "spearman",
  absolute = FALSE
)
print(plot_heatmap)

# Plot sample scores for the first latent component.
plot_scores = autoplot(
  gl_final,
  type = "mbspls_scores",
  source = "bootstrap",
  component = 1L,
  standardize = TRUE
)
print(plot_scores)

# Plot the score network when optional graph packages are installed.
if (requireNamespace("igraph", quietly = TRUE) &&
  requireNamespace("ggraph", quietly = TRUE)) {
  plot_network = autoplot(
    gl_final,
    type = "mbspls_network",
    source = "bootstrap",
    method = "spearman",
    cutoff = 0.25
  )
  print(plot_network)
}

# Check that the main fitted objects and plots exist.
stopifnot(inherits(gl_final, "GraphLearner"))
stopifnot(!is.null(gl_final$model))
stopifnot(is.matrix(c_matrix_final))
stopifnot(identical(external_eval$weights_source, "stable_ci"))
stopifnot(length(external_eval$ev_comp) == n_components)
stopifnot(inherits(plot_weights_raw, "ggplot"))
stopifnot(inherits(plot_weights_stable, "ggplot"))
stopifnot(inherits(plot_weight_ci, "ggplot"))
stopifnot(inherits(plot_variance, "ggplot"))
stopifnot(inherits(plot_scree, "ggplot"))
stopifnot(inherits(plot_heatmap, "ggplot"))
stopifnot(inherits(plot_scores, "ggplot"))

# End of analysis.
