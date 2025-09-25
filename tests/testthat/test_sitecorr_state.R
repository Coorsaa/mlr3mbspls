library(testthat)
library(mlr3)
library(mlr3pipelines)
library(mlr3mbspls)
library(checkmate)

generate_synthetic_data = function(n_samples = 200, n_sites = 4, seed = 42) {
  set.seed(seed)

  # Assertions for input validation
  assert_int(n_samples, lower = 50)
  assert_int(n_sites, lower = 2)

  # Create site labels
  sites = sample(paste0("Site_", seq_len(n_sites)), n_samples, replace = TRUE)

  # Generate synthetic multi-block data with site effects
  generate_block_data = function(n_samples, n_features, site_effect_strength = 2) {
    # Base signal
    base_data = matrix(rnorm(n_samples * n_features), nrow = n_samples)

    # Add site-specific batch effects
    site_effects = matrix(0, nrow = n_samples, ncol = n_features)
    for (i in seq_len(n_features)) {
      site_numeric = as.numeric(as.factor(sites))
      site_effects[, i] = site_effect_strength * sin(site_numeric * pi / 2) +
        rnorm(n_samples, 0, 0.5)
    }

    base_data + site_effects
  }

  # Generate different blocks
  clinical_data = generate_block_data(n_samples, 15, site_effect_strength = 5)
  genomic_data = generate_block_data(n_samples, 50, site_effect_strength = 20)
  proteomic_data = generate_block_data(n_samples, 25, site_effect_strength = 10)

  # Create column names
  setnames_list = list(
    clinical = paste0("clin_", seq_len(ncol(clinical_data))),
    genomic = paste0("gene_", seq_len(ncol(genomic_data))),
    proteomic = paste0("prot_", seq_len(ncol(proteomic_data)))
  )

  colnames(clinical_data) = setnames_list$clinical
  colnames(genomic_data) = setnames_list$genomic
  colnames(proteomic_data) = setnames_list$proteomic

  # Create covariates using data.table efficiently
  covariates = data.table(
    age = rnorm(n_samples, 50, 15),
    sex = factor(sample(c("M", "F"), n_samples, replace = TRUE)),
    bmi = rnorm(n_samples, 25, 5)
  )

  # Create outcome variable
  outcome = rowMeans(clinical_data[, 1:3]) + rowMeans(genomic_data[, 1:5]) +
    rnorm(n_samples, 0, 0.5)

  # Combine all data using data.table
  dt = data.table(
    site = sites,
    covariates,
    clinical_data,
    genomic_data,
    proteomic_data,
    outcome = outcome
  )

  # Return list with data and metadata
  list(
    data = dt,
    blocks = list(
      clinical = setnames_list$clinical,
      genomic = setnames_list$genomic,
      proteomic = setnames_list$proteomic
    ),
    n_samples = n_samples,
    n_sites = n_sites
  )
}

dt_info = generate_synthetic_data(n_samples = 100, n_sites = 3)
task = as_task_regr(dt_info$data, target = "outcome")
blocks = dt_info$blocks

methods = c("partial_corr", "combat", "dir")

for (m in methods) {
  po = po("sitecorr",
    blocks = blocks,
    site_col = "site",
    method = m,
    keep_site_col = TRUE)

  task_corr = po$train(list(task))[[1]]

  test_that(paste("state is correct for", m), {
    expect_true(po$is_trained)

    # Get the state from the public accessor
    st = po$state
    expect_equal(st$method, m)

    # Check expected names based on the method
    if (m == "partial_corr") {
      expect_true("beta" %in% names(st))
      expect_true("blocks" %in% names(st))
    } else if (m == "combat") {
      expect_true("combat" %in% names(st))
      expect_true("blocks" %in% names(st))
    } else if (m == "dir") {
      expect_true("dir" %in% names(st))
      expect_true("blocks" %in% names(st))
    }

    # basic dimensional check only for partial_corr
    if (m == "partial_corr") {
      for (bn in names(st$blocks)) {
        expect_equal(ncol(st$beta[[bn]]), length(st$blocks[[bn]]))
      }
    }
  })
}
