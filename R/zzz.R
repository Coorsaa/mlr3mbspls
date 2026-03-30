# nocov start
#' @importFrom mlr3 mlr_learners

.onLoad = function(libname, pkgname) { # nocov start
  add_or_replace = function(dict, key, value) {
    if (key %in% dict$keys()) {
      dict$remove(key)
    }
    dict$add(key, value)
  }

  add_or_replace(mlr3pipelines::mlr_pipeops, "sitecorr", PipeOpSiteCorrection)
  add_or_replace(mlr3pipelines::mlr_pipeops, "mbspls", PipeOpMBsPLS)
  add_or_replace(mlr3pipelines::mlr_pipeops, "mbspls_bootstrap_select", PipeOpMBsPLSBootstrapSelect)
  add_or_replace(mlr3pipelines::mlr_pipeops, "mbsplsxy", PipeOpMBsPLSXY)
  add_or_replace(mlr3pipelines::mlr_pipeops, "mbspca", PipeOpMBsPCA)
  add_or_replace(mlr3pipelines::mlr_pipeops, "blockscale", PipeOpBlockScaling)
  add_or_replace(mlr3pipelines::mlr_pipeops, "target_label_filter", PipeOpTargetLabelFilter)
  add_or_replace(mlr3pipelines::mlr_pipeops, "feature_suffix", PipeOpFeatureSuffix)

  add_or_replace(mlr3pipelines::mlr_graphs, "mbspls_preproc", mbspls_preproc_graph)
  add_or_replace(mlr3pipelines::mlr_graphs, "mbspls_graph", mbspls_graph)
  add_or_replace(mlr3pipelines::mlr_graphs, "mbspls_graph_learner", mbspls_graph_learner)
  add_or_replace(mlr3pipelines::mlr_graphs, "mbsplsxy_graph", mbsplsxy_graph)
  add_or_replace(mlr3pipelines::mlr_graphs, "mbsplsxy_graph_learner", mbsplsxy_graph_learner)
  add_or_replace(mlr3pipelines::mlr_graphs, "imputeknn", impute_knn_graph)

  add_or_replace(mlr3::mlr_measures, "mbspls.ev", MeasureMBsPLS_EV)
  add_or_replace(mlr3::mlr_measures, "mbspls.block_ev", MeasureMBsPLS_BlockEV)
  add_or_replace(mlr3::mlr_measures, "mbspls.mac", MeasureMBsPLS_MAC)
  add_or_replace(mlr3::mlr_measures, "mbspls.mac_evwt", MeasureMBsPLS_EVWeightedMAC)
  add_or_replace(mlr3::mlr_measures, "mbspca.mean_ev", MeasureMBSPCAMEV)

  add_or_replace(mlr3::mlr_learners, "regr.knngower", function() LearnerRegrKNNGower$new())
  add_or_replace(mlr3::mlr_learners, "classif.knngower", function() LearnerClassifKNNGower$new())

  add_or_replace(mlr3::mlr_tasks, "mbspls_synthetic_blocks", task_multiblock_synthetic(task_type = "clust"))
  add_or_replace(mlr3::mlr_tasks, "mbspls_synthetic_classif", task_multiblock_synthetic(task_type = "classif"))
  add_or_replace(mlr3::mlr_tasks, "mbspls_synthetic_regr", task_multiblock_synthetic(task_type = "regr"))
}
.onUnload = function(libpath) {
  # Remove PipeOps from mlr3pipelines dictionary
  x = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")

  pipeops_to_remove = c(
    "sitecorr",
    "mbspls",
    "mbspls_bootstrap_select",
    "mbsplsxy",
    "mbspca",
    "blockscale",
    "target_label_filter",
    "feature_suffix"
  )
  for (pipeop in pipeops_to_remove) {
    if (pipeop %in% x$keys()) {
      x$remove(pipeop)
    }
  }

  # Remove Graphs from mlr3pipelines dictionary
  g = utils::getFromNamespace("mlr_graphs", ns = "mlr3pipelines")
  graphs_to_remove = c("mbspls_preproc", "mbspls_graph", "mbspls_graph_learner", "mbsplsxy_graph", "mbsplsxy_graph_learner", "imputeknn")
  for (graph in graphs_to_remove) {
    if (graph %in% g$keys()) {
      g$remove(graph)
    }
  }

  # Remove measures from mlr3 dictionary
  y = utils::getFromNamespace("mlr_measures", ns = "mlr3")
  measures_to_remove = c("mbspls.ev", "mbspls.block_ev", "mbspls.mac", "mbspls.mac_evwt", "mbspca.mean_ev")
  for (measure in measures_to_remove) {
    if (measure %in% y$keys()) {
      y$remove(measure)
    }
  }

  # Remove learners from mlr3 dictionary
  z = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  learners_to_remove = c("regr.knngower", "classif.knngower")
  for (learner in learners_to_remove) {
    if (learner %in% z$keys()) {
      z$remove(learner)
    }
  }

  # Remove packaged tasks from mlr3 dictionary
  t = utils::getFromNamespace("mlr_tasks", ns = "mlr3")
  tasks_to_remove = c("mbspls_synthetic_blocks", "mbspls_synthetic_classif", "mbspls_synthetic_regr")
  for (task in tasks_to_remove) {
    if (task %in% t$keys()) {
      t$remove(task)
    }
  }
}
# nocov end
