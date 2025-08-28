# mlr3mbspls: Multi-Block Sparse PLS for mlr3

This package integrates multi-block sparse partial least squares (MB-sPLS) with the mlr3 machine learning framework. It provides custom PipeOp components for MB-sPLS projection and dimensionality reduction in multi-block data analysis workflows, with support for classification and clustering pipelines.

## Installation

```r
# Install from CRAN (once available)
install.packages("mlr3mbspls")

# Install development version from GitHub
# install.packages("devtools")
devtools::install_github("yourusername/mlr3mbspls")
```

## Features

- Custom `PipeOpMBsPLS` for dimensionality reduction and feature extraction from multi-block data
- Integrates with mlr3pipelines for building modular machine learning pipelines
- Support for preprocessing, supervised, and unsupervised analysis
- Dedicated unsupervised learner for exploratory multi-block analysis
- Visualization tools for examining latent variables and loadings

## Implementation

This package includes a complete implementation of the multi-block sparse PLS algorithm, requiring no external dependencies beyond standard R packages. All functionality is self-contained, making it easy to deploy in any environment that supports R and the mlr3 ecosystem.

## Basic Usage

```r
library(mlr3)
library(mlr3pipelines)
library(mlr3mbspls)

# Create multiblock task from data
task <- make_task_multiblock(
  list(block1 = X1, block2 = X2),
  target = y,
  task_type = "regr"
)

# Create MB-sPLS pipeline
mbpls_pipe <- po("mbspls", ncomp = 2, lambda = 0.5) %>>%
  po("learner", lrn("regr.ranger"))

# Create a graph learner
learner <- LearnerGraphMB$new(mbpls_pipe)

# Train and predict
learner$train(task)
prediction <- learner$predict(task)

# Get loadings and plot results
loadings <- learner$get_loadings(comp = 1)
learner$plot_latent_space(dims = c(1, 2))
```

### Unsupervised Analysis

```r
library(mlr3)
library(mlr3mbspls)

# Create multiblock task for unsupervised analysis
# Note: A target is still required by mlr3's task structure but won't be used for modeling
task <- make_task_multiblock(
  list(omics = X1, clinical = X2, imaging = X3),
  target = rep(1, nrow(X1)),  # Dummy target
  task_type = "regr"
)

# Create and configure the unsupervised learner
learner <- LearnerUnsupMBSPLS$new()
learner$param_set$values$ncomp <- 3
learner$param_set$values$lambda <- 0.7

# Train the model
learner$train(task)

# Analyze results
explained_var <- learner$get_explained_variance()
print(explained_var)  # Variance explained by each component

# Get top features
loadings <- learner$get_loadings(comp = 1, top = 10)
print(loadings)

# Visualize results
p1 <- learner$plot_latent_space(dims = c(1, 2))
p2 <- learner$plot_variable_importance(comp = 1, top = 5)
```

## Documentation

For more information, see the vignettes:

- [Quick Start Guide](vignettes/quickstart.html)
- [Full Pipeline Examples](vignettes/full_pipeline.html)

## License

GPL (>= 3)
