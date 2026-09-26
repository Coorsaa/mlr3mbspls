# Targeted null-behaviour regression for the fixed-weight, prediction-side
# permutation diagnostic. This checks the Monte Carlo calculation under a
# simple exchangeable null; it is not a validity proof for study pipelines.

suppressPackageStartupMessages(library(mlr3mbspls))

run_null_behavior = function(
  n_sim = 400L,
  n_perm = 199L,
  n = 40L,
  alpha = 0.05,
  seed = 20260830L
) {
  n_sim = as.integer(n_sim)
  n_perm = as.integer(n_perm)
  n = as.integer(n)
  # A rejection at alpha needs 1 / (n_perm + 1) <= alpha; otherwise the
  # type-I check is vacuous.
  minimum_permutations = as.integer(ceiling(1 / alpha)) - 1L
  if (is.na(n_sim) || n_sim < 100L || is.na(n) || n < 10L) {
    stop("`n_sim` must be >= 100 and `n` must be >= 10.", call. = FALSE)
  }
  if (is.na(n_perm) || n_perm < minimum_permutations) {
    stop(sprintf(
      "`n_perm` must be >= %d so that p-values can reach alpha = %.3f.",
      minimum_permutations, alpha
    ), call. = FALSE)
  }

  streams = mb_rng_streams(n_sim, seed)
  with_stream = getFromNamespace("with_rng_stream_local", "mlr3mbspls")
  permute_oos = getFromNamespace("cpp_perm_test_oos", "mlr3mbspls")

  p_values = vapply(seq_len(n_sim), function(i) {
    with_stream(streams[[i]], function() {
      x1 = matrix(stats::rnorm(n), ncol = 1L)
      x2 = matrix(stats::rnorm(n), ncol = 1L)
      result = permute_oos(
        X_test = list(block_1 = x1, block_2 = x2),
        W_trained = list(block_1 = 1, block_2 = 1),
        n_perm = n_perm,
        permute_all_blocks = FALSE
      )
      as.numeric(result$p_value)
    })
  }, numeric(1L))

  rejected = sum(p_values <= alpha)
  rate = rejected / n_sim
  z = stats::qnorm(0.975)
  denominator = 1 + z^2 / n_sim
  centre = (rate + z^2 / (2 * n_sim)) / denominator
  half_width = z * sqrt(
    rate * (1 - rate) / n_sim + z^2 / (4 * n_sim^2)
  ) / denominator
  lower = max(0, centre - half_width)
  upper = min(1, centre + half_width)
  minimum_possible = 1 / (n_perm + 1)

  result = data.frame(
    check = "fixed-weight permutation type-I rate",
    estimate = rate,
    lower_95 = lower,
    upper_95 = upper,
    target_or_bound = sprintf(
      "nominal %.3f; rate <= 0.100; p >= %.6f",
      alpha, minimum_possible
    ),
    n_sim = n_sim,
    n_perm = n_perm,
    rejected = rejected,
    zero_p_values = sum(p_values == 0),
    minimum_p_value = min(p_values),
    stringsAsFactors = FALSE
  )

  if (result$zero_p_values != 0L ||
    result$minimum_p_value + .Machine$double.eps < minimum_possible ||
    result$estimate > 0.10) {
    stop("Null-behaviour regression failed its pre-specified bounds.",
      call. = FALSE)
  }
  result
}

args = commandArgs(trailingOnly = TRUE)
n_sim = if (length(args) >= 1L) as.integer(args[[1L]]) else 400L
n_perm = if (length(args) >= 2L) as.integer(args[[2L]]) else 199L
print(run_null_behavior(n_sim = n_sim, n_perm = n_perm), row.names = FALSE)
