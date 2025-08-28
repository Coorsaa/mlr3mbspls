# mlr3mbspls: Multi-Block Sparse PLS for mlr3

[![R-CMD-check](https://github.com/coorsaa/mlr3mbspls/workflows/R-CMD-check/badge.svg)](https://github.com/coorsaa/mlr3mbspls/actions)
[![CRAN status](https://www.r-pkg.org/badges/version/mlr3mbspls)](https://CRAN.R-project.org/package=mlr3mbspls)

**mlr3mbspls** provides a complete implementation of **multi-block sparse partial least squares (MB-sPLS)** integrated with the mlr3 machine learning framework. This package enables comprehensive multi-omics and multi-block data analysis with advanced preprocessing, evaluation, and visualization capabilities.

## Key Features

### 🧬 **Multi-Block Analysis**

- **Native MB-sPLS implementation** with efficient C++/Armadillo backend
- **Block-wise scaling and preprocessing** for heterogeneous data types
- **Site effect correction** for multi-site studies (ComBat, limma, z-score methods)
- **Flexible sparsity control** per block and component

### 🔧 **mlr3 Integration**

- **Custom PipeOps**: `PipeOpMBsPLS`, `PipeOpBlockScaling`, `PipeOpSiteCorrection`
- **Specialized learners**: `LearnerClassifKNNGower`, `LearnerRegrKNNGower` with Gower distance
- **Advanced tuning**: `TunerSeqMBsPLS` for sequential component optimization
- **Custom measures**: MAC, explained variance, weighted correlations

### 📊 **Evaluation & Testing**

- **Permutation testing** for component significance
- **Bootstrap stability** selection and confidence intervals
- **Nested cross-validation** with `mbspls_nested_cv()`
- **Comprehensive evaluation measures** for unsupervised learning

### 📈 **Visualization**

- **8 different autoplot types** via `autoplot(GraphLearner)`
- **Weight plots**, variance explained, scree plots, correlation heatmaps
- **Network visualizations** and score plots
- **Bootstrap confidence intervals** and stability plots

## Installation

```r
# Install development version from GitHub
devtools::install_github("coorsaa/mlr3mbspls")

# Required dependencies
install.packages(c("mlr3", "mlr3pipelines", "mlr3cluster", "data.table", "ggplot2"))
```

## Quick Start

### Basic MB-sPLS Pipeline

```r
library(mlr3)
library(mlr3pipelines)
library(mlr3cluster)
library(mlr3mbspls)
library(data.table)

# Create multi-block data
set.seed(42)
n <- 200
clinical <- matrix(rnorm(n * 5), ncol = 5, dimnames = list(NULL, paste0("clinical_", 1:5)))
genomics <- matrix(rnorm(n * 20), ncol = 20, dimnames = list(NULL, paste0("gene_", 1:20)))
metabolomics <- matrix(rnorm(n * 15), ncol = 15, dimnames = list(NULL, paste0("metabol_", 1:15)))

# Combine into data.table
data_combined <- data.table(
  id = paste0("sample_", 1:n),
  clinical,
  genomics, 
  metabolomics
)

# Define block structure
blocks <- list(
  clinical = paste0("clinical_", 1:5),
  genomics = paste0("gene_", 1:20),
  metabolomics = paste0("metabol_", 1:15)
)

# Create clustering task
task <- TaskClust$new("multiblock", backend = data_combined, target = "id")
task$select(setdiff(task$feature_names, "id"))

# Create pipeline with MB-sPLS
graph <- po("scale") %>>%
  po("mbspls", 
     blocks = blocks, 
     ncomp = 3L, 
     performance_metric = "mac") %>>%
  po("learner", learner = lrn("clust.kmeans", centers = 3))

learner <- as_learner(graph)
learner$train(task)
prediction <- learner$predict(task)
```

### Advanced Pipeline with Site Correction

```r
# Add site information
data_with_sites <- copy(data_combined)
data_with_sites[, site := sample(c("site_A", "site_B", "site_C"), n, replace = TRUE)]
data_with_sites[, batch := sample(paste0("batch_", 1:5), n, replace = TRUE)]

# Create task with site variables
task_sites <- TaskClust$new("multiblock_sites", backend = data_with_sites, target = "id")

# Pipeline with preprocessing and site correction
preprocessing_graph <- 
  po("encode", method = "treatment") %>>%
  po("imputemedian") %>>%
  po("blockscaling", blocks = blocks, method = "unit_ssq") %>>%
  po("mbspls", 
     blocks = blocks,
     ncomp = 3L,
     performance_metric = "mac",
     permutation_test = TRUE,
     bootstrap_test = TRUE) %>>%
  po("learner", learner = lrn("clust.kmeans", centers = 3))

advanced_learner <- as_learner(preprocessing_graph)
advanced_learner$train(task)  # Use original task without sites for this example
```

## Comprehensive Visualization

The package provides extensive visualization capabilities through `autoplot()`:

```r
library(ggplot2)

# Train a model with logging for visualization
metrics_env <- new.env(parent = emptyenv())
viz_graph <- po("scale") %>>%
  po("mbspls", 
     blocks = blocks, 
     ncomp = 3L, 
     log_env = metrics_env,
     bootstrap_test = TRUE) %>>%
  po("learner", learner = lrn("clust.kmeans", centers = 3))

viz_learner <- as_learner(viz_graph)
viz_learner$train(task)

# Feature importance (top weights per component/block)
autoplot(viz_learner, type = "mbspls_weights", top_n = 5)

# Variance explained per block and component
autoplot(viz_learner, type = "mbspls_variance", show_total = TRUE)

# Scree plot (objective function per component)
autoplot(viz_learner, type = "mbspls_scree")

# Correlation heatmap of latent variables
autoplot(viz_learner, type = "mbspls_heatmap", method = "spearman")

# Network plot (requires igraph/ggraph)
autoplot(viz_learner, type = "mbspls_network", cutoff = 0.3)

# Score plots for specific components
autoplot(viz_learner, type = "mbspls_scores", component = 1)

# Evaluate on new data
test_task <- task$clone()$filter(1:50)  # subset for demo
autoplot(viz_learner, type = "mbspls_variance", new_task = test_task)
```

## Advanced Features

### Sequential Component Tuning

```r
# Use specialized tuner for component-wise optimization
tuner <- TunerSeqMBsPLS$new()

# Create tuning instance with MB-sPLS specific measure
instance <- ti(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measures = msr("mbspls.mac"),
  terminator = trm("evals", n_evals = 50)
)

# Optimize (conceptual - requires full setup)
# tuner$optimize(instance)
```

### Custom Measures for Evaluation

```r
# Specialized measures for MB-sPLS evaluation
measure_mac <- MeasureMBsPLS_MAC$new()              # Mean absolute correlation
measure_ev <- MeasureMBsPLS_EV$new()                # Explained variance  
measure_evwt <- MeasureMBsPLS_EVWeightedMAC$new()   # EV-weighted MAC

# Use in resampling
measures <- list(measure_mac, measure_ev, measure_evwt)
```

### Nested Cross-Validation

```r
# Robust evaluation with nested CV
nested_results <- mbspls_nested_cv(
  task = task,
  graphlearner = learner,
  rs_outer = rsmp("cv", folds = 5),
  rs_inner = rsmp("cv", folds = 3),
  ncomp = 5L,
  tuner_budget = 20L,
  performance_metric = "mac"
)
```

## Custom Learners

The package includes custom learners optimized for multi-block data:

```r
# k-NN with Gower distance (good for mixed-type data)
knn_gower <- lrn("classif.knngower", k = 5, predict_type = "prob")

# Use in classification pipeline after MB-sPLS
classif_graph <- po("mbspls", blocks = blocks, ncomp = 3L) %>>%
  po("learner", learner = knn_gower)
```

## Key Components

| Component                  | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `PipeOpMBsPLS`           | Main MB-sPLS transformation with sparsity control  |
| `PipeOpBlockScaling`     | Block-wise scaling (unit SSQ, z-score, feature SD) |
| `PipeOpSiteCorrection`   | Site effect correction (ComBat, limma, z-score)    |
| `TunerSeqMBsPLS`         | Sequential component-wise hyperparameter tuning    |
| `LearnerClassifKNNGower` | k-NN classification with Gower distance            |
| `MeasureMBsPLS_*`        | Specialized evaluation measures                    |
| `autoplot.GraphLearner`  | Comprehensive visualization suite                  |

## Use Cases

**mlr3mbspls** is particularly well-suited for:

- **Multi-omics integration** (genomics, proteomics, metabolomics)
- **Neuroimaging studies** with multiple modalities
- **Multi-site clinical studies** requiring batch correction
- **High-dimensional data** with block structure
- **Unsupervised exploratory analysis** of complex datasets
- **Feature selection** across heterogeneous data types

## Implementation Details

- **Native C++ backend** using Rcpp/RcppArmadillo for performance
- **Memory efficient** algorithms with deflation-based component extraction
- **Extensive testing** with permutation and bootstrap validation
- **Full mlr3 ecosystem integration** with pipelines, tuning, and resampling
- **Comprehensive documentation** with practical examples

## Documentation

For detailed examples and tutorials:

- [**Quick Start Guide**](vignettes/quickstart.html) - Complete introduction with examples
- Package documentation: `help(package = "mlr3mbspls")`
- Function reference: `?PipeOpMBsPLS`, `?autoplot.GraphLearner`

## Citation

If you use mlr3mbspls in your research, please cite:

```
@Manual{mlr3mbspls,
  title = {mlr3mbspls: Multi-Block Sparse PLS for mlr3},
  author = {Stefan Coors and Clara Vetter},
  year = {2025},
  note = {R package version 0.2.0},
  url = {https://github.com/coorsaa/mlr3mbspls}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPL (>= 3)
