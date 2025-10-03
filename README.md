<div align="center">

# mlr3mbspls: Multi-Block Sparse PLS for mlr3

[![R-CMD-check](https://github.com/coorsaa/mlr3mbspls/workflows/R-CMD-check/badge.svg)](https://github.com/coorsaa/mlr3mbspls/actions)
[![CRAN status](https://www.r-pkg.org/badges/version/mlr3mbspls)](https://CRAN.R-project.org/package=mlr3mbspls)

</div>

`mlr3mbspls` integrates **multi-block sparse partial least squares (MB-sPLS)** with the mlr3 ecosystem: pipelines, tuning, resampling, custom measures, rich visualisations, bootstrap stability selection, prediction‑side validation and nested CV utilities. A high‑performance C++/Armadillo backend powers the core algorithms (training + test EV, permutation, bootstrap, sparsity by block/component, deflation).

---
## 🔑 Highlights

### Multi-Block Representation Learning
* Sequential orthogonal MB‑sPLS with per‑block L¹ sparsity (vector or full `c_matrix`)
* Two optimisation targets: mean absolute correlation (MAC) or Frobenius norm
* Training‑time permutation early stopping (per component)
* Prediction‑side validation: permutation or bootstrap inference on latent correlation
* Block‑wise explained variance (EV) + per‑component EV on train & test

### Pipeline Components (PipeOps)
* `PipeOpMBsPLS` – main transformer (produces per‑block latent scores `LVk_block`)
* `PipeOpMBsPLSBootstrapSelect` – post‑hoc bootstrap feature & component selection (CI or frequency method) with component re‑numbering
* `PipeOpMBsPCA` – multi‑block sparse PCA analogue
* `PipeOpMBsPLSXY` – supervised XY variant
* `PipeOpBlockScaling` – unit sum‑of‑squares or feature‑wise scaling / z‑scoring (optionally divide by √p)
* `PipeOpSiteCorrection` – multi‑block site/batch correction (methods defined per site variable)
* `PipeOpFeatureSuffix` – systematic feature renaming
* `PipeOpTargetLabelFilter` – target label filtering convenience op

### Learners & Imputation Helpers
* `LearnerClassifKNNGower`, `LearnerRegrKNNGower` – kNN using Gower distance for mixed types
* `impute_knn_graph()` – two‑step numeric/factor kNN imputation graph using above learners

### Tuning & Orchestration
* `TunerSeqMBsPLS`, `TunerSeqMBsPCA` – sequential component‑wise tuning (progressively add components)
* Sparse hyper‑parameters exposed with consistent `c_<block>` naming or full `c_matrix`

### Evaluation & Stability Tooling
* Measures: `MeasureMBsPLS_MAC`, `MeasureMBsPLS_EV`, `MeasureMBsPLS_BlockEV`, `MeasureMBsPLS_EVWeightedMAC`, `MeasureMBSPCAMEV`
* `compute_test_ev()`, `compute_pipeop_test_ev()` – EV + objective on new data
* `mbspls_flip_weights()` – deterministic reorientation (sign alignment)
* `mbspls_extract_bootstrap_means()` – summarise bootstrap runs
* `mbspls_plot_block_weight_ci()` – block weight CIs
* Aggregation helpers: `aggregate_mbspls_payloads()`, `collect_mbspls_nested_cv()`

### Higher Level Graph Utilities
* `mbspls_preproc_graph()` – canonical preprocessing (type conversion → encoding → kNN impute → site correction → scaling)
* `mbspls_graph_learner()` – end‑to‑end GraphLearner constructor (preproc → MB‑sPLS → optional bootstrap selection → downstream learner)

### Resampling & Batch Infrastructure
* `mbspls_nested_cv()` – nested CV (inner tuning budget + outer evaluation)
* `mbspls_nested_cv_batchtools()` – batchtools backend variant

### Visualisation (S3 Autoplot on `GraphLearner`)
Types include: weights (raw / stability‑filtered), variance, scree, correlation heatmap, network, scores, block EV trajectories, bootstrap diagnostics.

---
## 📦 Installation

```r
# Development version
devtools::install_github("coorsaa/mlr3mbspls")

# Core dependencies (install if missing)
install.packages(c("mlr3","mlr3pipelines","mlr3cluster","data.table","ggplot2"))
```

Optional: network plots require `igraph` + `ggraph`.

---
## 🚀 Quick Start (Unsupervised Multi-Block Latent Space)

```r
library(mlr3)
library(mlr3pipelines)
library(mlr3cluster)
library(mlr3mbspls)
library(mlr3viz)
library(data.table)

set.seed(42)
n = 200
clinical = matrix(rnorm(n * 5),  ncol = 5,  dimnames = list(NULL, paste0("clinical_", 1:5)))
genomics = matrix(rnorm(n * 20), ncol = 20, dimnames = list(NULL, paste0("gene_", 1:20)))
metabol = matrix(rnorm(n * 15),  ncol = 15, dimnames = list(NULL, paste0("metabol_", 1:15)))

dt = data.table(id = paste0("sample_", seq_len(n)), clinical, genomics, metabol)

blocks = list(
  clinical = grep("^clinical_", names(dt), value = TRUE),
  genomics = grep("^gene_", names(dt), value = TRUE),
  metabol  = grep("^metabol_", names(dt), value = TRUE)
)

task = TaskClust$new("mb", backend = dt)
task$select(setdiff(task$feature_names, "id"))

graph = po("blockscale", param_vals = list(blocks = blocks, method = "unit_ssq")) %>>%
  po("mbspls", blocks = blocks, ncomp = 3L, performance_metric = "mac") %>>%
  po("learner", learner = lrn("clust.kmeans", centers = 3))

gl = as_learner(graph)
gl$train(task)
pred = gl$predict(task)

autoplot(gl, type = "mbspls_weights", source = "weights", top_n = 10)
autoplot(gl, type = "mbspls_variance", show_total = TRUE)
autoplot(gl, type = "mbspls_heatmap", method = "spearman", absolute = FALSE)
```

---
## 🧪 Prediction-Side Validation & Bootstrap Selection

```r
log_env = new.env(parent = emptyenv())

graph_sel = po("blockscale", param_vals = list(blocks = blocks)) %>>%
  po("mbspls", blocks = blocks, ncomp = 4L, performance_metric = "mac",
     permutation_test = TRUE, n_perm = 200L, perm_alpha = 0.05,
     val_test = "permutation", val_test_n = 500L, val_test_alpha = 0.05,
     append = TRUE,               # expose upstream LV columns to selection op
     store_train_blocks = TRUE,   # pass original blocks for bootstrap
     log_env = log_env) %>>%
  po("mbspls_bootstrap_select", log_env = log_env, bootstrap = TRUE,
     B = 200L, selection_method = "ci", align = "block_sign") %>>%
  po("learner", learner = lrn("clust.kmeans", centers = 3))

gl_sel = as_learner(graph_sel)
gl_sel$train(task)

# Stable (post-selection) latent columns now in the task representation
gl_sel$model$mbspls_bootstrap_select$kept_blocks_per_comp
```

---
## 🛠 Higher Level Convenience Graph

```r
# --- Site / batch effect correction example ---
# PipeOpSiteCorrection supports per-block methods: "partial_corr", "combat", "dir".
# For "combat" supply a list with elements site=<char1>, covariates=<char_vec>.
# For "partial_corr" supply a character vector of (site + optional covariates) columns.

# Add mock site / batch / covariate columns to the data (if not already present)
dt[, site  := sample(c("S1","S2","S3"), .N, TRUE)]
dt[, batch := sample(c("B1","B2"),   .N, TRUE)]
dt[, age   := rnorm(.N, 50, 8)]
dt[, sex   := sample(c("F","M"), .N, TRUE)]

# Update task backend to include new columns
task = TaskClust$new("mb", backend = dt)
task$select(setdiff(task$feature_names, "id"))

# Per-block site correction specifications
site_correction = list(
  clinical = list(site = "site", covariates = c("age","sex")), # ComBat with covariates
  genomics = c("batch"),                                          # partial correlation on batch
  metabol  = "site"                                               # single categorical site (partial_corr)
)

# Corresponding methods per block
site_correction_methods = list(
  clinical = "combat",
  genomics = "partial_corr",
  metabol  = "partial_corr"
)

gl_full = mbspls_graph_learner(
  blocks = blocks,
  site_correction = site_correction,
  site_correction_methods = site_correction_methods,
  keep_site_col = FALSE,      # drop site / covariate columns after correction
  ncomp = 3L,
  performance_metric = "mac",
  permutation_test = TRUE,
  n_perm = 200L,
  bootstrap = TRUE,
  B = 100L,
  selection_method = "frequency",
  frequency_threshold = 0.1
)

gl_full$train(task)
```

---
## 📊 Visualisation Examples

```r
library(mlr3viz)
autoplot(gl_sel, type = "mbspls_weights", source = "weights", top_n = 10)
autoplot(gl_sel, type = "mbspls_weights", source = "bootstrap", alpha_by_stability = TRUE)
autoplot(gl_sel, type = "mbspls_variance", show_total = TRUE)
autoplot(gl_sel, type = "mbspls_heatmap", method = "spearman", absolute = FALSE)
# Optional network (needs igraph/ggraph installed)
# autoplot(gl_sel, type = "mbspls_network", cutoff = 0.1)
```

`mbspls_plot_block_weight_ci()` produces per‑block weight confidence intervals after bootstrap selection:

```r
mbspls_plot_block_weight_ci(gl_sel, source = "bootstrap", alpha_by_stability = TRUE)
```

---
## 📏 Measures

| Measure Class | Purpose |
| ------------- | ------- |
| `MeasureMBsPLS_MAC` | Mean absolute correlation of block scores |
| `MeasureMBsPLS_EV` | Total explained variance (summed blocks) |
| `MeasureMBsPLS_BlockEV` | Per‑block EV matrix access |
| `MeasureMBsPLS_EVWeightedMAC` | MAC weighted by EV contribution |
| `MeasureMBSPCAMEV` | EV (multi‑block sparse PCA) |

Use like any mlr3 measure:

```r
ms = list(msr("mbspls.mac"), msr("mbspls.ev"))
rr = resample(task, gl, rsmp("cv", folds = 3), store_models = TRUE, measures = ms)
rr$aggregate()
```

---
## 🔄 Nested Cross-Validation

```r
library(mlr3tuning)
res_nested = mbspls_nested_cv(
  task = task,
  graphlearner = gl_full,
  rs_outer = rsmp("cv", folds = 3),
  rs_inner = rsmp("cv", folds = 2),
  ncomp = 4L,
  tuner_budget = 10L,
  performance_metric = "mac"
)
str(res_nested)
```

Batchtools version (for HPC) is available via `mbspls_nested_cv_batchtools()`.

---
## 🔧 Sequential Component Tuning

```r
tuner = TunerSeqMBsPLS$new()
instance = ti(
  task = task,
  learner = gl_full,
  resampling = rsmp("cv", folds = 2),
  measures = msr("mbspls.mac"),
  terminator = trm("evals", n_evals = 20)
)
# tuner$optimize(instance)
# instance$result$learner_param_vals[[1]]$c_matrix
```

---
## 🔍 Useful Low-Level Helpers

| Function | Role |
| -------- | ---- |
| `compute_test_ev()` | Compute EV + objective on new matrices (standalone) |
| `compute_pipeop_test_ev()` | Same for a PipeOp state representation |
| `mbspls_eval_new_data()` | Score new data given training state (matrix interface) |
| `mbspls_flip_weights()` | Sign alignment (GraphLearner, PipeOp, list) |
| `mbspls_extract_bootstrap_means()` | Summarise bootstrap weight means |
| `aggregate_mbspls_payloads()` | Merge logged payloads (e.g. across resamples) |
| `collect_mbspls_nested_cv()` | Collect nested CV payload archives |

---
## 🧠 Custom Learners (Gower kNN)

```r
knn_cls = lrn("classif.knngower", k = 5)
knn_reg = lrn("regr.knngower", k = 5)
```

These are used implicitly inside `impute_knn_graph()` and can be part of supervised pipelines downstream of MB‑sPLS/MB‑sPCA representations.

---
## 🧷 Reproducible Sparsity Specification

Two options:

1. Per‑block constraints automatically created: parameters named `c_<block>` with default upper bound √p.
2. Provide a `c_matrix` (rows = blocks, cols = components) – overrides `ncomp` and per‑block `c_` values.

```r
graph_cmat = po("mbspls", blocks = blocks, c_matrix = matrix(c(2,2,3,3,1,1), nrow = 3, byrow = TRUE))
```

---
## 📄 Vignette

See the Quickstart vignette for an end‑to‑end multi‑omics example:

```r
vignette("quickstart", package = "mlr3mbspls")
```

---
## 📚 Citation

If you use `mlr3mbspls` in academic work please cite:

```
@Manual{mlr3mbspls,
  title = {mlr3mbspls: Multi-Block Sparse PLS for mlr3},
  author = {Stefan Coors and Clara Sophie Vetter},
  year = {2025},
  note = {R package version 0.2.6},
  url = {https://github.com/coorsaa/mlr3mbspls}
}
```

---
## 🤝 Contributing

Issues & PRs welcome. Please open an issue for substantial interface changes before implementing. Run `pre-commit` hooks + `R CMD check` locally.

---
## 📜 License

GPL (>= 3)
