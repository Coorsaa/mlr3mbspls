# nocov start
#' @importFrom mlr3 mlr_learners

.onLoad = function(libname, pkgname) { # nocov start
  mlr3pipelines::mlr_pipeops$add("sitecorr", PipeOpSiteCorrection)
  mlr3pipelines::mlr_pipeops$add("mbspls", PipeOpMBsPLS)
  mlr3pipelines::mlr_pipeops$add("mbspls_bootstrap_select", PipeOpMBsPLSBootstrapSelect)
  mlr3pipelines::mlr_pipeops$add("mbsplsxy", PipeOpMBsPLSXY)
  mlr3pipelines::mlr_pipeops$add("mbspca", PipeOpMBsPCA)
  mlr3pipelines::mlr_pipeops$add("blockscale", PipeOpBlockScaling)
  mlr3pipelines::mlr_pipeops$add("target_label_filter", PipeOpTargetLabelFilter)
  mlr3pipelines::mlr_pipeops$add("feature_suffix", PipeOpFeatureSuffix)

  mlr3pipelines::mlr_graphs$add("mbspls_preproc", mbspls_preproc_graph)
  mlr3pipelines::mlr_graphs$add("mbspls_graph_learner", mbspls_graph_learner)
  mlr3pipelines::mlr_graphs$add("imputeknn", impute_knn_graph)

  mlr3::mlr_measures$add("mbspls.ev", MeasureMBsPLS_EV)
  mlr3::mlr_measures$add("mbspls.block_ev", MeasureMBsPLS_BlockEV)
  mlr3::mlr_measures$add("mbspls.mac", MeasureMBsPLS_MAC)
  mlr3::mlr_measures$add("mbspls.mac_evwt", MeasureMBsPLS_EVWeightedMAC)
  mlr3::mlr_measures$add("mbspca.mean_ev", MeasureMBSPCAMEV)

  mlr3::mlr_learners$add("regr.knngower", function() LearnerRegrKNNGower$new())
  mlr3::mlr_learners$add("classif.knngower", function() LearnerClassifKNNGower$new())

  # a discoverable default graph: site correction ➜ MB‑sPLS ➜ dummy learner
  # mlr3pipelines::mlr_graphs$add("mbspls_default", graph_mbspls)
}
.onUnload = function(libpath) {
  # Remove PipeOps from mlr3pipelines dictionary
  x = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")

  pipeops_to_remove = c("sitecorr", "mbspls", "mbsplsxy", "mbspca", "blockscale")
  for (pipeop in pipeops_to_remove) {
    if (pipeop %in% x$keys()) {
      x$remove(pipeop)
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
}
# nocov end
