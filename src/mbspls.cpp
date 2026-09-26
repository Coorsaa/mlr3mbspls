// =====================================================================
//  File: src/mbspls.cpp
//  -------------------------------------------------------------------
//  C++ / Armadillo implementation of unsupervised multi-block sparse PLS
//  (MB-sPLS) with permutation-test based early stopping.
//  Functions exported to R:
//    • cpp_mbspls_one_lv()              - one-component solver
//    • cpp_mbspls_multi_lv()            - multi-component solver
//    • cpp_mbspls_multi_lv_cmatrix()    - multi-component solver with a
//                                         per-component sparsity matrix
//  The one-component solver uses Gauss-Seidel block updates from a
//  deterministic cross-covariance start, so fits do not depend on the RNG.
//  The multi-component routines assume column-centred blocks: the weights are
//  invariant to column shifts, but the deflation loadings p_b = X_b' t_b /
//  t_b't_b, the explained variances (relative to sum-of-squares) and the
//  scores are not. The R callers centre each block by its training means.
// =====================================================================
#define ARMA_DONT_ALIGN_MEMORY
#include <RcppArmadillo.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>  // std::pair

using arma::uvec;
using arma::vec;
using arma::mat;
using std::size_t;

// ─────────────────────────────────────────────────────────────────────
//  VALIDATION FUNCTIONS
// ─────────────────────────────────────────────────────────────────────
inline void log_info(const std::string&) {}

inline bool is_valid_matrix(const arma::mat& X) {
  if (X.n_rows == 0 || X.n_cols == 0) return false;
  if (!X.is_finite()) return false;
  if (arma::accu(arma::abs(X)) < 1e-12) return false;
  return true;
}

inline bool is_valid_vector(const arma::vec& v) {
  if (v.n_elem == 0) return false;
  if (!v.is_finite()) return false;
  return true;
}

// ─────────────────────────────────────────────────────────────────────
//  CORE COMPUTATION FUNCTIONS - Internal use only
// ─────────────────────────────────────────────────────────────────────

struct ScoreMatrix {
  arma::mat T;
  std::vector<bool> valid_blocks;
  int n_valid;
  ScoreMatrix(int n, int B) : T(n, B, arma::fill::zeros), valid_blocks(B, false), n_valid(0) {}
};

ScoreMatrix compute_scores_core(const std::vector<arma::mat>& X,
                                const std::vector<arma::vec>& W) {
  const int B = static_cast<int>(X.size());
  const int n = (B > 0) ? static_cast<int>(X[0].n_rows) : 0;
  ScoreMatrix result(n, B);

  for (int b = 0; b < B; ++b) {
    if (is_valid_matrix(X[b]) && is_valid_vector(W[b]) && X[b].n_cols == W[b].n_elem) {
      arma::vec t = X[b] * W[b];
      if (is_valid_vector(t)) {
        double v = arma::var(t);
        if (std::isfinite(v) && v > 1e-12) {
          result.T.col(b) = t;
          result.valid_blocks[b] = true;
          result.n_valid++;
        }
      }
    }
  }
  return result;
}

// CORE: Rank ties using average ranks (1-based)
arma::vec rank_ties_avg(const arma::vec& x) {
  arma::uvec ord = arma::sort_index(x);         // ascending order
  arma::vec r(x.n_elem, arma::fill::zeros);
  arma::uword i = 0;
  while (i < x.n_elem) {
    arma::uword j = i;
    // find tie block [i, j]
    while (j + 1 < x.n_elem && x(ord(j+1)) == x(ord(i))) ++j;
    double avg = 0.5 * (i + j) + 1.0;           // average 1-based rank
    for (arma::uword k = i; k <= j; ++k) r(ord(k)) = avg;
    i = j + 1;
  }
  return r;
}

// CORE: Standardized correlation computation
double compute_correlation_core(const arma::vec& x, const arma::vec& y, bool spearman=false) {
  if (!is_valid_vector(x) || !is_valid_vector(y) || x.n_elem != y.n_elem) {
    Rcpp::stop("compute_correlation_core: correlation requires two finite vectors of equal length.");
  }
  try {
    if (spearman) {
      arma::vec rx = rank_ties_avg(x);
      arma::vec ry = rank_ties_avg(y);
      return arma::as_scalar(arma::cor(rx, ry));
    } else {
      return arma::as_scalar(arma::cor(x, y));
    }
  } catch (const std::exception& e) {
    Rcpp::stop(std::string("compute_correlation_core: correlation failed: ") + e.what());
  } catch (...) {
    Rcpp::stop("compute_correlation_core: correlation failed with an unknown error.");
  }
}

// --------------------------------------------------------------------
//  Average absolute correlation  ⟨|r|⟩   (bounded 0…1)
// --------------------------------------------------------------------
double compute_block_objective_core(const ScoreMatrix& scores,
                                    bool spearman = false,
                                    bool frobenius = false)
{
  const int B = scores.T.n_cols;
  double acc = 0.0;
  int valid_pairs = 0;
  for (int i = 0; i < B - 1; ++i)
    if (scores.valid_blocks[i])
      for (int j = i + 1; j < B; ++j)
        if (scores.valid_blocks[j]) {
          double r = compute_correlation_core(scores.T.col(i),
                                              scores.T.col(j),
                                              spearman);
          if (std::isfinite(r)) {
            acc += frobenius ? r * r : std::abs(r);
            ++valid_pairs;
          }
        }
  if (!valid_pairs) {
    Rcpp::stop("compute_block_objective_core: no valid block pairs remain; the latent-correlation objective is undefined.");
  }
  return frobenius ? std::sqrt(acc)          // ‖R‖_F
                   : acc / valid_pairs;      // ⟨|r|⟩
}

// CORE: Alternative direct computation (for when you have X and W)
double compute_objective_direct_core(const std::vector<arma::mat>& X,
                                     const std::vector<arma::vec>& W,
                                     bool spearman = false,
                                     bool frobenius = false) {
  ScoreMatrix scores = compute_scores_core(X, W);
  return compute_block_objective_core(scores, spearman, frobenius);
}

// [[Rcpp::export]]
double cpp_block_objective_oos(const Rcpp::List& X_blocks,
                               const Rcpp::List& W_list,
                               bool              spearman  = false,
                               bool              frobenius = false)
{
  const int B = X_blocks.size();
  if (B < 2) {
    Rcpp::stop("cpp_block_objective_oos: need at least two blocks to compute an inter-block objective.");
  }
  if (W_list.size() != B) {
    Rcpp::stop("W_list must have the same length as X_blocks.");
  }

  int n = -1;
  std::vector<arma::vec> T(B);
  std::vector<unsigned char> valid(B, 0);
  int n_valid = 0;

  for (int b = 0; b < B; ++b) {
    Rcpp::NumericMatrix Xr = X_blocks[b];
    Rcpp::NumericVector Wr = W_list[b];

    const int nb = Xr.nrow();
    const int pb = Xr.ncol();

    if (n < 0) {
      n = nb;
    } else if (nb != n) {
      Rcpp::stop("All blocks must have the same number of rows.");
    }

    if (nb < 1 || pb < 1 || Wr.size() < 1) {
      Rcpp::stop(std::string("cpp_block_objective_oos: block ") + std::to_string(b + 1) +
                 " has empty data or an empty weight vector.");
    }

    arma::mat X(Xr.begin(), nb, pb, false, true);
    if (!X.is_finite()) {
      Rcpp::stop(std::string("cpp_block_objective_oos: block ") + std::to_string(b + 1) +
                 " contains non-finite values.");
    }
    if (Wr.size() != pb) {
      Rcpp::stop(std::string("cpp_block_objective_oos: weight vector length mismatch in block ") +
                 std::to_string(b + 1) + ": expected " + std::to_string(pb) +
                 ", got " + std::to_string(Wr.size()) + ".");
    }

    arma::vec w(Wr.begin(), pb, false, true);
    if (!w.is_finite()) {
      Rcpp::stop(std::string("cpp_block_objective_oos: weight vector for block ") + std::to_string(b + 1) +
                 " contains non-finite values.");
    }

    arma::vec tb = X * w;
    if (!tb.is_finite() || tb.n_elem < 2) {
      Rcpp::stop(std::string("cpp_block_objective_oos: score computation failed for block ") + std::to_string(b + 1) + ".");
    }

    const double v = arma::var(tb);
    if (!(std::isfinite(v) && v > 1e-12)) {
      Rcpp::stop(std::string("cpp_block_objective_oos: block score variance is numerically zero for block ") +
                 std::to_string(b + 1) + ".");
    }
    T[b] = std::move(tb);
    valid[b] = 1;
    ++n_valid;
  }

  if (n_valid < 2) {
    Rcpp::stop("cpp_block_objective_oos: fewer than two valid block scores remain; the inter-block objective is undefined.");
  }

  auto fast_pearson = [](const arma::vec& x, const arma::vec& y) {
    const double mx = arma::mean(x);
    const double my = arma::mean(y);
    const arma::vec xc = x - mx;
    const arma::vec yc = y - my;
    const double sxx = arma::dot(xc, xc);
    const double syy = arma::dot(yc, yc);
    if (!(std::isfinite(sxx) && std::isfinite(syy)) || sxx <= 1e-12 || syy <= 1e-12) {
      Rcpp::stop("cpp_block_objective_oos: Pearson correlation is undefined because at least one score vector has zero variance.");
    }
    return arma::dot(xc, yc) / std::sqrt(sxx * syy);
  };

  double acc = 0.0;
  int valid_pairs = 0;
  for (int i = 0; i < B - 1; ++i) {
    if (!valid[i]) continue;
    for (int j = i + 1; j < B; ++j) {
      if (!valid[j]) continue;
      const double r = spearman ? compute_correlation_core(T[i], T[j], true) : fast_pearson(T[i], T[j]);
      if (std::isfinite(r)) {
        acc += frobenius ? (r * r) : std::abs(r);
        ++valid_pairs;
      }
    }
  }

  if (valid_pairs == 0) {
    Rcpp::stop("cpp_block_objective_oos: no valid block pairs remain for the requested objective.");
  }
  return frobenius ? std::sqrt(acc) : (acc / valid_pairs);
}

// CORE: Build target score for one-LV updates
arma::vec build_target_score_core(const ScoreMatrix& scores, int exclude_block) {
  const int n = scores.T.n_rows;
  const int B = scores.T.n_cols;

  arma::vec target(n, arma::fill::zeros);
  int used = 0;

  for (int b = 0; b < B; ++b) {
    if (b == exclude_block || !scores.valid_blocks[b]) continue;

    arma::vec t = scores.T.col(b);
    double mu = arma::mean(t);
    double sd = std::sqrt(arma::var(t));
    if (!std::isfinite(sd) || sd < 1e-12) continue;

    target += (t - mu) / sd;   // z-score
    ++used;
  }
  if (used > 0) target /= used;
  return target;               // may be zero if no usable blocks
}

// Enhanced block-wise deflation with validation
inline bool deflate_block(arma::mat&      X_b,
                         const arma::vec& t_b,
                         const arma::vec& p_b)
{
  if (!is_valid_matrix(X_b) || !is_valid_vector(t_b) || !is_valid_vector(p_b)) {
    return false;
  }
  
  double t_norm_sq = arma::dot(t_b, t_b);
  if (t_norm_sq < 1e-12) {
    return false;
  }
  
  arma::mat deflation = t_b * p_b.t();
  if (!deflation.is_finite()) {
    return false;
  }
  
  X_b -= deflation;
  return is_valid_matrix(X_b);
}


// ─────────────────────────────────────────────────────────────────────
//  SOFT-THRESHOLDING UTILITIES
// ─────────────────────────────────────────────────────────────────────

// --- helper: plain soft threshold (no extra scaling) ---
inline arma::vec soft_no_scale(const arma::vec& g, double alpha) {
  return arma::sign(g) % arma::max(arma::abs(g) - alpha,
                                   arma::zeros<arma::vec>(g.n_elem));
}

// Shared by MB-sPLS and MB-sPCA. Solve max g'w subject to ||w||2 = 1
// and ||w||1 <= c, including the tied-maximum case where soft-thresholding
// alone cannot reach the requested L1 budget.
arma::vec pmd_update_bisection(const arma::vec& g,
                              double c,
                              int maxit = 80,
                              double tol = 1e-8)
{
  const arma::uword p = g.n_elem;
  if (!p || !g.is_finite()) {
    Rcpp::stop("pmd_update_bisection: gradient must be finite and non-empty.");
  }
  if (!std::isfinite(c) || c < 1.0) {
    Rcpp::stop("pmd_update_bisection: L1 budget must be finite and at least 1.");
  }
  const double magnitude = arma::abs(g).max();
  if (!std::isfinite(magnitude) || magnitude < 1e-16) {
    Rcpp::stop("pmd_update_bisection: gradient norm is numerically zero.");
  }

  const arma::vec scaled = g / magnitude;
  arma::vec weights = scaled / arma::norm(scaled, 2);
  if (arma::accu(arma::abs(weights)) <= c) return weights;

  const arma::uvec maxima = arma::find(arma::abs(scaled) == 1.0);
  const double m = static_cast<double>(maxima.n_elem);
  weights.zeros();
  if (c <= std::sqrt(m)) {
    // Put all mass on tied maxima. The largest coefficient and remaining
    // equal coefficients satisfy both norm constraints and attain max(|g|)*c.
    const double spread = std::sqrt(std::max(0.0, m - c * c));
    const double first = (c + std::sqrt(m - 1.0) * spread) / m;
    const double rest = m > 1.0 ? (c - spread / std::sqrt(m - 1.0)) / m : 0.0;
    for (arma::uword j = 0; j < maxima.n_elem; ++j) {
      const arma::uword at = maxima(j);
      weights(at) = (scaled(at) < 0.0 ? -1.0 : 1.0) * (j == 0 ? first : rest);
    }
    return weights;
  }

  // Keep the feasible side of the bracket, so the returned vector never
  // exceeds its budget even when the iteration limit is reached.
  for (arma::uword j = 0; j < maxima.n_elem; ++j) {
    weights(maxima(j)) = scaled(maxima(j)) / std::sqrt(m);
  }
  double lo = 0.0, hi = 1.0;
  for (int it = 0; it < maxit; ++it) {
    const double threshold = 0.5 * (lo + hi);
    arma::vec candidate = soft_no_scale(scaled, threshold);
    const double candidate_norm = arma::norm(candidate, 2);
    if (candidate_norm == 0.0) {
      hi = threshold;
      continue;
    }
    candidate /= candidate_norm;
    const double l1 = arma::accu(arma::abs(candidate));
    if (l1 > c) {
      lo = threshold;
    } else {
      weights = candidate;
      hi = threshold;
      if (c - l1 <= tol) break;
    }
  }
  return weights;
}


// ─────────────────────────────────────────────────────────────────────
//  ONE-LV SOLVER CORE
// ─────────────────────────────────────────────────────────────────────

// Power iterations used by the deterministic start. They only need a good
// basin, not a converged singular vector, so the cap keeps the start cheap
// relative to the Gauss-Seidel solve that follows. Caps from 10 to 300 gave
// the same final objectives in simulations, and 10 had the lowest total cost.
#ifndef MBSPLS_INIT_MAX_ITER
#define MBSPLS_INIT_MAX_ITER 10
#endif

#ifndef MBSPLS_INIT_TOL
#define MBSPLS_INIT_TOL 1e-6
#endif

// Fixed dense start vector for the power iterations (splitmix64 sequence).
// It never touches R's RNG. A data column or an all-ones vector can be
// orthogonal to the leading singular vector for structured data (independent
// feature groups, mixed-sign loadings); a generic dense vector is not.
inline arma::vec fixed_start_vector(arma::uword p) {
  arma::vec v(p);
  std::uint64_t state = 0x2545F4914F6CDD1DULL;
  for (arma::uword j = 0; j < p; ++j) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    v(j) = static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5;
  }
  return v / arma::norm(v, 2);
}

inline arma::vec centered_scores(const arma::mat& Xb, const arma::vec& w) {
  arma::vec t = Xb * w;
  t -= arma::mean(t);
  return t;
}

// Leading eigenvector of a positive semi-definite operator by power
// iteration from `v`. Returns an empty vector when the operator maps the
// current iterate to (numerically) zero relative to `scale`.
template <typename Operator>
arma::vec leading_psd_direction(Operator apply, arma::vec v, double scale) {
  for (int it = 0; it < MBSPLS_INIT_MAX_ITER; ++it) {
    arma::vec next = apply(v);
    const double magnitude = arma::norm(next, 2);
    if (!std::isfinite(magnitude) || magnitude <= 1e-12 * scale) {
      return arma::vec();
    }
    next /= magnitude;
    const double change = arma::norm(next - v, 2);
    v = std::move(next);
    if (change < MBSPLS_INIT_TOL) break;
  }
  return v;
}

// Start for block b: leading left singular vector of the centred
// cross-covariance X_b' [X_c / ||X_c||_F]_{c != b}. Scaling each other block
// to unit total variance mirrors the solver's target, in which every other
// block contributes one z-scored score regardless of its size. The operator
// v -> X_b' sum_c X_c X_c' X_b v / ||X_c||_F^2 runs through the n-dimensional
// score space, so no p_b x sum(p_c) matrix is formed. Falls back to the
// block's leading principal axis when the cross-covariance is numerically
// zero.
arma::vec mbspls_start_direction(const std::vector<arma::mat>& X,
                                 const arma::vec& ss_centred,
                                 int b) {
  const int B = static_cast<int>(X.size());
  const arma::mat& Xb = X[b];
  const arma::uword p = Xb.n_cols;
  const double ss_b = ss_centred(b);
  if (!(std::isfinite(ss_b) && ss_b > 1e-12)) {
    Rcpp::stop(std::string("cpp_mbspls_one_lv: cannot initialise block ") + std::to_string(b + 1) +
               " because its centred data are numerically zero.");
  }
  int n_other = 0;
  for (int c = 0; c < B; ++c) {
    if (c != b && ss_centred(c) > 1e-12) ++n_other;
  }

  // The fixed start can be orthogonal to the range of a low-rank operator, so
  // a vanishing first step is retried from the unit vector of the block's
  // highest-variance column before the operator is declared degenerate. For
  // the principal-axis operator that start never vanishes when ss_b > 0.
  const arma::uword j_star = arma::index_max(arma::var(Xb, 0, 0));
  arma::vec e_star(p, arma::fill::zeros);
  e_star(j_star) = 1.0;
  const std::vector<arma::vec> starts = {fixed_start_vector(p), e_star};

  arma::vec v;
  if (n_other > 0) {
    const auto cross_covariance = [&](const arma::vec& w) {
      const arma::vec u = centered_scores(Xb, w);
      arma::vec y(u.n_elem, arma::fill::zeros);
      for (int c = 0; c < B; ++c) {
        if (c == b || !(ss_centred(c) > 1e-12)) continue;
        const arma::vec yc = X[c].t() * u;
        y += centered_scores(X[c], yc) / ss_centred(c);
      }
      return arma::vec(Xb.t() * y);
    };
    for (const arma::vec& start : starts) {
      v = leading_psd_direction(cross_covariance, start, ss_b * n_other);
      if (!v.is_empty()) return v;
    }
  }

  const auto principal_axis = [&](const arma::vec& w) {
    return arma::vec(Xb.t() * centered_scores(Xb, w));
  };
  for (const arma::vec& start : starts) {
    v = leading_psd_direction(principal_axis, start, ss_b);
    if (!v.is_empty()) return v;
  }

  Rcpp::stop(std::string("cpp_mbspls_one_lv: cannot initialise block ") + std::to_string(b + 1) +
             " because its centred data are numerically zero.");
}

// Deterministic start: project each block's start direction onto the
// sparsity constraint set, then orient block scores coherently so that
// oppositely signed blocks cannot cancel in the first cross-block target.
std::vector<arma::vec> mbspls_initial_weights(const std::vector<arma::mat>& X,
                                              const arma::vec& c_constraints) {
  const int B = static_cast<int>(X.size());
  const arma::uword n = X[0].n_rows;
  arma::vec ss_centred(B);
  for (int b = 0; b < B; ++b) {
    ss_centred(b) = (n - 1.0) * arma::accu(arma::var(X[b], 0, 0));
  }
  std::vector<arma::vec> W(B);
  arma::vec consensus(n, arma::fill::zeros);
  for (int b = 0; b < B; ++b) {
    W[b] = pmd_update_bisection(mbspls_start_direction(X, ss_centred, b), c_constraints(b));
    const arma::vec t = centered_scores(X[b], W[b]);
    const double sd = arma::norm(t, 2);
    if (!std::isfinite(sd) || sd < 1e-12) continue;
    if (arma::dot(t, consensus) < 0.0) {
      W[b] = -W[b];
      consensus -= t / sd;
    } else {
      consensus += t / sd;
    }
  }
  return W;
}

// Refresh one block's score column and validity after its weight update,
// using the same variance rule as compute_scores_core(). Sparse weights only
// touch their support columns.
inline void refresh_block_score(ScoreMatrix& scores,
                                const arma::mat& Xb,
                                const arma::vec& wb,
                                int b) {
  const arma::uvec support = arma::find(wb);
  const arma::vec t = (2 * support.n_elem < wb.n_elem)
    ? arma::vec(Xb.cols(support) * wb.elem(support))
    : arma::vec(Xb * wb);
  const double v = arma::var(t);
  const bool ok = t.is_finite() && std::isfinite(v) && v > 1e-12;
  if (ok) {
    scores.T.col(b) = t;
  } else {
    scores.T.col(b).zeros();
  }
  if (ok != scores.valid_blocks[b]) {
    scores.n_valid += ok ? 1 : -1;
    scores.valid_blocks[b] = ok;
  }
}

// Global sign convention: flip all blocks jointly (never individually, which
// would change the relative block orientation) so that the largest-magnitude
// weight of the first block is positive; ties resolve to the lowest index.
inline void canonicalise_global_sign(std::vector<arma::vec>& W) {
  if (W.empty() || W[0].is_empty()) return;
  const arma::uword at = arma::index_max(arma::abs(W[0]));
  if (W[0](at) < 0.0) {
    for (auto& w : W) w = -w;
  }
}

struct OneLvFit {
  std::vector<arma::vec> W;
  bool converged;
  int iterations;
};

// One-component MB-sPLS by block-coordinate (Gauss-Seidel) sweeps. Each
// block is updated towards the mean z-scored score of the other blocks at
// their latest values, and its own score is refreshed before the next block's
// target is built. Convergence is declared on the weights,
// max_b ||w_b - w_b_prev||_2 < tol, not on the sign-invariant objective.
// Every caller (the exported one-LV routine, the multi-LV routines and the
// permutation refits) uses this function, so they share the deterministic
// start and the stopping rule.
OneLvFit mbspls_fit_one_lv(const std::vector<arma::mat>& X,
                           const arma::vec& c_constraints,
                           int max_iter,
                           double tol) {
  const int B = static_cast<int>(X.size());
  if (B < 2) Rcpp::stop("cpp_mbspls_one_lv: at least 2 blocks are required to define a cross-block latent variable; got " + std::to_string(B) + ".");
  if (static_cast<int>(c_constraints.n_elem) != B) Rcpp::stop("c_constraints length must match number of blocks");
  if (max_iter < 1) Rcpp::stop("cpp_mbspls_one_lv: max_iter must be at least 1.");
  if (!std::isfinite(tol) || tol <= 0.0) Rcpp::stop("cpp_mbspls_one_lv: tol must be finite and positive.");
  if (!c_constraints.is_finite() || arma::any(c_constraints < 1.0)) {
    Rcpp::stop("cpp_mbspls_one_lv: every sparsity constraint must be finite and at least 1.");
  }
  for (int b = 1; b < B; ++b) {
    if (X[b].n_rows != X[0].n_rows) Rcpp::stop("Inconsistent sample sizes across blocks");
  }
  if (X[0].n_rows < 3) Rcpp::stop("Need at least 3 samples");

  OneLvFit fit;
  fit.W = mbspls_initial_weights(X, c_constraints);
  fit.converged = false;
  fit.iterations = 0;

  ScoreMatrix scores = compute_scores_core(X, fit.W);

  for (int it = 0; it < max_iter; ++it) {
    double max_change = 0.0;

    for (int b = 0; b < B; ++b) {
      arma::vec target = build_target_score_core(scores, b);

      if (arma::norm(target, 2) < 1e-12) {
        Rcpp::stop(std::string("cpp_mbspls_one_lv: no valid cross-block target score could be formed for block ") + std::to_string(b + 1) + ". Check that at least two blocks contain informative, non-degenerate signals.");
      }

      arma::vec grad = X[b].t() * target;

      if (!is_valid_vector(grad)) {
        Rcpp::stop(std::string("cpp_mbspls_one_lv: gradient became non-finite for block ") + std::to_string(b + 1) + ".");
      }

      // Enforce L2 = 1 and L1 <= c via bisection (PMD update)
      arma::vec w_new = pmd_update_bisection(grad, c_constraints(b));
      max_change = std::max(max_change, arma::norm(w_new - fit.W[b], 2));
      fit.W[b] = std::move(w_new);
      refresh_block_score(scores, X[b], fit.W[b], b);
    }

    fit.iterations = it + 1;
    if (max_change < tol) {
      log_info("     one-LV solver converged after " + std::to_string(it + 1) + " iterations");
      fit.converged = true;
      break;
    }
  }

  canonicalise_global_sign(fit.W);
  return fit;
}


// ─────────────────────────────────────────────────────────────────────
//  EXPORTED FUNCTIONS - R Interface (using core functions internally)
// ─────────────────────────────────────────────────────────────────────

// The fit is a deterministic function of the data and settings; it does not
// draw from R's RNG.
// [[Rcpp::export(rng = false)]]
Rcpp::List cpp_mbspls_one_lv(const Rcpp::List&  X_blocks,
                             const arma::vec&   c_constraints,
                             int                max_iter,
                             double             tol,
                             bool               frobenius = false,
                             bool               spearman = false)
{
  log_info("↳  cpp_mbspls_one_lv()");

  const int B = X_blocks.size();

  if (B < 2) Rcpp::stop("cpp_mbspls_one_lv: at least 2 blocks are required to define a cross-block latent variable; got " + std::to_string(B) + ".");

  // Keep the (possibly coerced) R matrices protected for the whole call; the
  // Armadillo views below do not own their memory.
  std::vector<Rcpp::NumericMatrix> X_keep;
  std::vector<arma::mat> X;
  X_keep.reserve(B);
  X.reserve(B);
  arma::uword n = 0;

  for (int b = 0; b < B; ++b) {
    X_keep.push_back(Rcpp::NumericMatrix(X_blocks[b]));
    Rcpp::NumericMatrix& Xr = X_keep.back();
    X.emplace_back(Xr.begin(), Xr.nrow(), Xr.ncol(), false, true);
    if (!is_valid_matrix(X.back())) {
      Rcpp::stop("Invalid matrix in block " + std::to_string(b + 1));
    }

    if (b == 0) {
      n = X.back().n_rows;
    } else if (X.back().n_rows != n) {
      Rcpp::stop("Inconsistent sample sizes across blocks");
    }

    if (n < 3) Rcpp::stop("Need at least 3 samples");
  }

  const OneLvFit fit = mbspls_fit_one_lv(X, c_constraints, max_iter, tol);

  // Build final results using core computation
  const ScoreMatrix final_scores = compute_scores_core(X, fit.W);
  const double final_objective =
    compute_block_objective_core(final_scores, spearman, frobenius);

  Rcpp::List W_out(B);
  for (int b = 0; b < B; ++b) {
    W_out[b] = fit.W[b];
  }

  return Rcpp::List::create(
    Rcpp::_["W"]          = W_out,
    Rcpp::_["T_mat"]      = final_scores.T,
    Rcpp::_["objective"]  = final_objective,
    Rcpp::_["converged"]  = fit.converged,
    Rcpp::_["iterations"] = fit.iterations
  );
}

// [[Rcpp::export]]
double perm_test_component(
    const std::vector<arma::mat> &X_orig,
    const std::vector<arma::vec> &W_orig,
    const arma::vec             &c_vec,
    int                           n_perm   = 1000,
    bool                          spearman = false,
    int                           max_iter = 500,
    double                        tol      = 1e-4,
    double                        early_stop_threshold = 0.05,
    bool                          frobenius = false
  )
{
  const int B = static_cast<int>(X_orig.size());
  if (B < 2) return 1.0;
  if (static_cast<int>(W_orig.size()) != B ||
      static_cast<int>(c_vec.n_elem) != B) {
    Rcpp::stop("perm_test_component requires one weight vector and one sparsity constraint per block.");
  }
  if (n_perm < 1) Rcpp::stop("perm_test_component requires at least one sampled permutation.");
  if (max_iter < 1) Rcpp::stop("perm_test_component requires max_iter >= 1.");
  if (!std::isfinite(tol) || tol <= 0.0) Rcpp::stop("perm_test_component requires finite, positive tol.");
  if (!c_vec.is_finite() || arma::any(c_vec < 1.0)) {
    Rcpp::stop("perm_test_component requires finite sparsity constraints of at least 1.");
  }
  const int n = static_cast<int>(X_orig[0].n_rows);
  if (n < 3) Rcpp::stop("perm_test_component requires at least 3 rows per block.");
  for (int b = 0; b < B; ++b) {
    if (!is_valid_matrix(X_orig[b]) || !is_valid_vector(W_orig[b]) ||
        X_orig[b].n_rows != static_cast<arma::uword>(n) ||
        X_orig[b].n_cols != W_orig[b].n_elem) {
      Rcpp::stop(std::string("perm_test_component received an invalid or dimensionally incompatible block/weight pair at block ") + std::to_string(b + 1) + ".");
    }
  }
  (void)early_stop_threshold;  // Retained for API compatibility; full B is used.

  // Observed statistic (same pipeline as permutations, no alignment)
  const double obj_ref = compute_objective_direct_core(X_orig, W_orig, spearman, frobenius);

  int ge = 0;

  for (int p = 0; p < n_perm; ++p) {
    // ────────────────────────────────────────────────────────────────
    // Permute ALL blocks independently to break cross-block alignment
    // ────────────────────────────────────────────────────────────────
    std::vector<arma::mat> X = X_orig;
    for (int b = 0; b < B; ++b) {
      arma::uvec idx = arma::shuffle(arma::regspace<arma::uvec>(0, n - 1));
      X[b] = X[b].rows(idx);
    }

    // Refit one LV to the permuted blocks with the SAME penalties, start and
    // stopping rule as the observed fit
    try {
      const OneLvFit fit = mbspls_fit_one_lv(X, c_vec, max_iter, tol);

      // Evaluate the SAME statistic on permuted fit — NO rotations/Procrustes
      const double obj_perm = compute_objective_direct_core(X, fit.W, spearman, frobenius);

      if (obj_perm >= obj_ref - 100.0 * std::numeric_limits<double>::epsilon() * std::abs(obj_ref)) ++ge;

    } catch (const std::exception &e) {
      Rcpp::stop(std::string("Permutation replicate ") + std::to_string(p + 1) +
                 " failed while fitting a one-component MB-sPLS model: " + e.what());
    }
  }

  // Add-one smoothing (same convention you used before)
  return (ge + 1.0) / (n_perm + 1.0);
}


// [[Rcpp::export]]
Rcpp::List cpp_mbspls_multi_lv(const Rcpp::List&  X_blocks,
                               const arma::vec&   c_constraints,
                               int                K        = 2,
                               int                max_iter = 500,
                               double             tol      = 1e-4,
                               bool               spearman = false,
                               bool               do_perm  = false,
                               int                n_perm   = 100,
                               double             alpha    = 0.05,
                               bool               frobenius = false)
{
  log_info("📦  cpp_mbspls_multi_lv() called");

  if (K < 1) Rcpp::stop("K must be >= 1");

  const int B = X_blocks.size();
  if (B < 2) Rcpp::stop("cpp_mbspls_multi_lv: at least 2 blocks are required; got " + std::to_string(B) + ".");

  std::vector<arma::mat> X(B);
  for (int b = 0; b < B; ++b) {
    arma::mat Xi = Rcpp::as<arma::mat>(X_blocks[b]); // may be an external view
    X[b] = arma::mat(Xi);
    if (!is_valid_matrix(X[b])) {
      Rcpp::stop("Invalid input matrix in block " + std::to_string(b + 1));
    }
  }
  const int n = X[0].n_rows;

  arma::vec ss_tot(B, arma::fill::zeros);
  for (int b = 0; b < B; ++b)
    ss_tot(b) = arma::accu(arma::square(X[b]));

  std::vector<std::vector<arma::vec>> W_all, P_all;
  arma::mat  T_all(n, 0);
  std::vector<double> obj_vec, p_vec;
  std::vector<arma::vec> ev_block_list;
  std::vector<double>   ev_comp_list;
  std::vector<bool>     converged_list;
  std::vector<int>      iterations_list;

  for (int k = 0; k < K; ++k) {
    log_info("⏩  extracting component " + std::to_string(k + 1));

    bool all_valid = true;
    for (int b = 0; b < B; ++b) {
      if (!is_valid_matrix(X[b])) {
        all_valid = false;
        log_info("Block " + std::to_string(b + 1) + " invalid, stopping extraction");
        break;
      }
    }
    if (!all_valid) break;

    OneLvFit fit;
    try {
      fit = mbspls_fit_one_lv(X, c_constraints, max_iter, tol);
    } catch (const std::exception &e) {
      Rcpp::stop(std::string("Component extraction failed at component ") + std::to_string(k + 1) + ": " + e.what());
    }

    const std::vector<arma::vec>& Wk = fit.W;

    // CORE: Use standardized objective computation
    double obj_k = compute_objective_direct_core(X, Wk, spearman, frobenius);
    log_info("     objective = " + std::to_string(obj_k));

    double p_val = NA_REAL;
    bool keep_it = true;
    if (do_perm) {
      log_info("     permutation test (" + std::to_string(n_perm) + " reps)");
      try {
        p_val = perm_test_component(X, Wk, c_constraints,
                            n_perm, spearman, max_iter, tol,
                            /* early_stop_threshold */ alpha,
                            /* frobenius */            frobenius);
        keep_it = (p_val <= alpha);
        log_info("     p-value = " + std::to_string(p_val));
      } catch (const std::exception &e) {
        Rcpp::stop(std::string("Permutation test failed at component ") + std::to_string(k + 1) + ": " + e.what());
      }
    }

    // CORE: Use standardized score computation
    ScoreMatrix scores_k = compute_scores_core(X, Wk);
    arma::mat Tk = scores_k.T;
    std::vector<arma::vec> Pk(B);
    arma::vec ev_block(B, arma::fill::zeros);
    double ss_exp_total = 0.0;

    for (int b = 0; b < B; ++b) {
      arma::vec tb = Tk.col(b);
      if (!is_valid_vector(tb)) continue;
      
      double tb_norm_sq = arma::dot(tb, tb);
      if (tb_norm_sq < 1e-12) continue;
      
      arma::vec pb = X[b].t() * tb / tb_norm_sq;
      if (!is_valid_vector(pb)) continue;
      
      double ss_exp = tb_norm_sq * arma::dot(pb, pb);
      ev_block(b) = ss_exp / ss_tot(b);
      ss_exp_total += ss_exp;

      Pk[b] = pb;
    }

    double ev_comp = ss_exp_total / arma::accu(ss_tot);

    bool keep_component = (!do_perm) || keep_it || k == 0;   // always keep LV-1

    if (!keep_component) {           // non-significant and NOT LV-1
      log_info("🚦  LV not significant - stopping extraction");
      break;                         // nothing stored yet
    }

    /* ---------- store the component ---------- */
    T_all = arma::join_rows(T_all, Tk);
    W_all.push_back(Wk);
    P_all.push_back(Pk);
    obj_vec.push_back(obj_k);
    p_vec.push_back(p_val);
    ev_block_list.push_back(ev_block);
    ev_comp_list.push_back(ev_comp);
    converged_list.push_back(fit.converged);
    iterations_list.push_back(fit.iterations);

    /* ---------- stop right after LV-1 if it was not significant ---------- */
    if (do_perm && !keep_it) {       // this can only be k == 0
      log_info("🚦  LV-1 kept but not significant - no further extraction");
      break;
    }

    /* ---------- deflate only when another component is requested ---------- */
    if (k + 1 < K) {
      for (int b = 0; b < B; ++b) {
        if (!deflate_block(X[b], Tk.col(b), Pk[b])) {
          Rcpp::stop(std::string("cpp_mbspls_multi_lv: deflation failed for block ") +
                     std::to_string(b + 1) + ", component " + std::to_string(k + 1) + ".");
        }
      }
    }
  }

  const size_t nc = W_all.size();

  Rcpp::List W_out(nc), P_out(nc);
  arma::mat  ev_block_mat(nc, B);
  arma::vec  ev_comp_vec(nc);

  for (size_t k = 0; k < nc; ++k) {
    Rcpp::List Wi(B), Pi(B);
    for (int b = 0; b < B; ++b) {
      Wi[b] = W_all[k][b];
      Pi[b] = P_all[k][b];
    }
    W_out[k] = Wi;
    P_out[k] = Pi;
    ev_block_mat.row(k) = ev_block_list[k].t();
    ev_comp_vec(k) = ev_comp_list[k];
  }

  return Rcpp::List::create(
    Rcpp::_["W"]          = W_out,
    Rcpp::_["P"]          = P_out,
    Rcpp::_["T_mat"]      = T_all,
    Rcpp::_["objective"]  = obj_vec,
    Rcpp::_["p_values"]   = p_vec,
    Rcpp::_["ev_block"]   = ev_block_mat,
    Rcpp::_["ev_comp"]    = ev_comp_vec,
    Rcpp::_["converged"]  = Rcpp::wrap(converged_list),
    Rcpp::_["iterations"] = Rcpp::wrap(iterations_list)
  );
}


// [[Rcpp::export]]
Rcpp::List cpp_mbspls_multi_lv_cmatrix(const Rcpp::List&  X_blocks,
                                       const arma::mat&   c_matrix,
                                       int                max_iter   = 500,
                                       double             tol        = 1e-4,
                                       bool               spearman   = false,
                                       bool               do_perm    = false,
                                       int                n_perm     = 100,
                                       double             alpha      = 0.05,
                                       bool               frobenius  = false)
{
  log_info("📦  cpp_mbspls_multi_lv_cmatrix() called");

  const int B = X_blocks.size();
  if (B == 0) Rcpp::stop("Empty X_blocks list");
  if (static_cast<int>(c_matrix.n_rows) != B)
    Rcpp::stop("c_matrix must have %d rows (blocks); got %d",
               B, static_cast<int>(c_matrix.n_rows));
  const int K = static_cast<int>(c_matrix.n_cols);
  if (K < 1) Rcpp::stop("c_matrix must have at least one column (components)");

  // Materialise blocks and basic checks
  std::vector<arma::mat> X(B);
  for (int b = 0; b < B; ++b) {
    arma::mat Xi = Rcpp::as<arma::mat>(X_blocks[b]);
    X[b] = arma::mat(Xi);
    if (!is_valid_matrix(X[b])) {
      Rcpp::stop("Invalid input matrix in block %d", b + 1);
    }
  }
  const int n = static_cast<int>(X[0].n_rows);

  // Pre-compute total SSqs for EVs
  arma::vec ss_tot(B, arma::fill::zeros);
  for (int b = 0; b < B; ++b) ss_tot(b) = arma::accu(arma::square(X[b]));
  const double ss_all = arma::accu(ss_tot);

  // Holders
  std::vector<std::vector<arma::vec>> W_all, P_all;
  arma::mat  T_all(n, 0);
  std::vector<double> obj_vec, p_vec;
  std::vector<arma::vec> ev_block_list;
  std::vector<double>   ev_comp_list;
  std::vector<bool>     converged_list;
  std::vector<int>      iterations_list;

  for (int k = 0; k < K; ++k) {
    log_info("⏩  extracting component " + std::to_string(k + 1));

    // Current sparsity vector
    arma::vec c_vec = c_matrix.col(k);
    if (static_cast<int>(c_vec.n_elem) != B)
      Rcpp::stop("Unexpected c_matrix column length at component %d", k + 1);

    // Fit one LV on the *current* (already-deflated up to k-1) blocks
    OneLvFit fit;
    try {
      fit = mbspls_fit_one_lv(X, c_vec, max_iter, tol);
    } catch (const std::exception &e) {
      Rcpp::stop(std::string("Component extraction failed at component ") + std::to_string(k + 1) + ": " + e.what());
    }

    const std::vector<arma::vec>& Wk = fit.W;

    // Objective for bookkeeping
    const double obj_k = compute_objective_direct_core(X, Wk, spearman, frobenius);

    // Optional permutation test with early stopping on alpha
    double p_val = NA_REAL;
    bool   keep_it = true;
    if (do_perm) {
      try {
        p_val = perm_test_component(
          /* X_orig  */ X,
          /* W_orig  */ Wk,
          /* c_vec   */ c_vec,
          /* n_perm  */ n_perm,
          /* spearman*/ spearman,
          /* max_iter*/ max_iter,
          /* tol     */ tol,
          /* early   */ alpha,       // <-- wire alpha for early stop
          /* frob    */ frobenius
        );
        keep_it = (p_val <= alpha);
      } catch (const std::exception &e) {
        Rcpp::stop(std::string("Permutation test failed at component ") + std::to_string(k + 1) + ": " + e.what());
      }
    }

    // Scores for this component
    ScoreMatrix scores_k = compute_scores_core(X, Wk);
    arma::mat Tk = scores_k.T;
    // Loadings, EVs, and deflation
    std::vector<arma::vec> Pk(B);
    arma::vec ev_block(B, arma::fill::zeros);
    double ss_exp_total = 0.0;

    for (int b = 0; b < B; ++b) {
      arma::vec tb = Tk.col(b);
      if (!is_valid_vector(tb)) continue;

      const double tb_norm_sq = arma::dot(tb, tb);
      if (tb_norm_sq < 1e-12) continue;

      arma::vec pb = X[b].t() * tb / tb_norm_sq;
      if (!is_valid_vector(pb)) continue;

      const double ss_exp = tb_norm_sq * arma::dot(pb, pb);
      if (ss_tot(b) > 1e-12) ev_block(b) = ss_exp / ss_tot(b);
      ss_exp_total += ss_exp;

      Pk[b] = pb;
    }

    double ev_comp = 0.0;
    if (ss_all > 1e-12) ev_comp = std::max(0.0, std::min(1.0, ss_exp_total / ss_all));

    // Keep LV-1 regardless of significance; otherwise stop if not significant
    const bool keep_component = (!do_perm) || keep_it || k == 0;
    if (!keep_component) {
      log_info("🚦  LV not significant - stopping extraction");
      break; // nothing has been stored for this k
    }

    // Store results
    T_all = arma::join_rows(T_all, Tk);
    W_all.push_back(Wk);
    P_all.push_back(Pk);
    obj_vec.push_back(obj_k);
    p_vec.push_back(p_val);
    ev_block_list.push_back(ev_block);
    ev_comp_list.push_back(ev_comp);
    converged_list.push_back(fit.converged);
    iterations_list.push_back(fit.iterations);

    // If LV-1 was not significant: keep it but stop afterwards
    if (do_perm && !keep_it) {
      log_info("🚦  LV-1 kept but not significant - no further extraction");
      break;
    }

    // Deflate only when another component will be extracted.
    if (k + 1 < K) {
      for (int b = 0; b < B; ++b) {
        if (!deflate_block(X[b], Tk.col(b), Pk[b])) {
          Rcpp::stop(std::string("cpp_mbspls_multi_lv_cmatrix: deflation failed for block ") +
                     std::to_string(b + 1) + ", component " + std::to_string(k + 1) + ".");
        }
      }
    }
  }

  // Pack return
  const std::size_t nc = W_all.size();
  Rcpp::List W_out(nc), P_out(nc);
  arma::mat ev_block_mat(nc, B);
  arma::vec ev_comp_vec(nc);

  for (std::size_t k = 0; k < nc; ++k) {
    Rcpp::List Wi(B), Pi(B);
    for (int b = 0; b < B; ++b) {
      Wi[b] = W_all[k][b];
      Pi[b] = P_all[k][b];
    }
    W_out[k] = Wi;
    P_out[k] = Pi;
    ev_block_mat.row(k) = ev_block_list[k].t();
    ev_comp_vec(k)      = ev_comp_list[k];
  }

  return Rcpp::List::create(
    Rcpp::_["W"]          = W_out,
    Rcpp::_["P"]          = P_out,
    Rcpp::_["T_mat"]      = T_all,
    Rcpp::_["objective"]  = obj_vec,
    Rcpp::_["p_values"]   = p_vec,
    Rcpp::_["ev_block"]   = ev_block_mat,
    Rcpp::_["ev_comp"]    = ev_comp_vec,
    Rcpp::_["converged"]  = Rcpp::wrap(converged_list),
    Rcpp::_["iterations"] = Rcpp::wrap(iterations_list)
  );
}


// [[Rcpp::export]]
Rcpp::List cpp_compute_test_ev_core(const Rcpp::List& X_blocks_test,
                                    const Rcpp::List& W_all,
                                    const Rcpp::List& P_all,
                                    bool              deflate            = true,
                                    bool              spearman           = false,
                                    bool              frobenius          = false,
                                    double            eps_var            = 1e-12,
                                    bool              use_train_loadings = true,
                                    int               clamp_mode         = 0)
{
  const int B = X_blocks_test.size();
  const int K = W_all.size();

  if (B < 1 || K < 1) {
    return Rcpp::List::create(
      Rcpp::_["ev_block"] = arma::mat(),
      Rcpp::_["ev_comp"] = arma::vec(),
      Rcpp::_["ev_block_cum"] = arma::mat(),
      Rcpp::_["ev_comp_cum"] = arma::vec(),
      Rcpp::_["mac_comp"] = arma::vec(),
      Rcpp::_["valid_block"] = Rcpp::LogicalMatrix(0, 0),
      Rcpp::_["T_mat"] = arma::mat()
    );
  }

  std::vector<arma::mat> X_base(B);
  std::vector<arma::mat> X_work(B);

  int n_test = -1;
  for (int b = 0; b < B; ++b) {
    X_base[b] = Rcpp::as<arma::mat>(X_blocks_test[b]);
    if (n_test < 0) n_test = static_cast<int>(X_base[b].n_rows);
    if (static_cast<int>(X_base[b].n_rows) != n_test) {
      Rcpp::stop("All test blocks must have the same number of rows.");
    }
    X_work[b] = X_base[b];
  }

  arma::vec ss_tot_test(B, arma::fill::zeros);
  for (int b = 0; b < B; ++b) {
    ss_tot_test(b) = arma::accu(X_base[b] % X_base[b]);
  }
  const double ss_tot_all = arma::accu(ss_tot_test);

  auto clamp_ratio = [&](double x) {
    if (!std::isfinite(x)) {
      Rcpp::stop("cpp_compute_test_ev_core: encountered a non-finite explained-variance ratio.");
    }
    if (clamp_mode == 1) return x < 0.0 ? 0.0 : x;            // zero
    if (clamp_mode == 2) return std::max(0.0, std::min(1.0, x)); // zero_one
    return x;                                                   // none
  };

  arma::mat ev_block_inc(K, B, arma::fill::zeros);
  arma::mat ev_block_cum(K, B, arma::fill::zeros);
  arma::vec ev_comp_inc(K, arma::fill::zeros);
  arma::vec ev_comp_cum(K, arma::fill::zeros);
  arma::vec mac_comp(K, arma::fill::zeros);
  arma::umat valid_block(K, B, arma::fill::zeros);
  arma::mat T_mat(n_test, K * B, arma::fill::zeros);

  arma::vec ss_exp_cum_block(B, arma::fill::zeros);
  double ss_exp_cum_total = 0.0;

  for (int k = 0; k < K; ++k) {
    Rcpp::List Wk = Rcpp::as<Rcpp::List>(W_all[k]);
    Rcpp::List Pk;
    if (use_train_loadings && P_all.size() > k) {
      Pk = Rcpp::as<Rcpp::List>(P_all[k]);
    }

    arma::mat Tk(n_test, B, arma::fill::zeros);

    for (int b = 0; b < B; ++b) {
      const arma::mat& Xb = deflate ? X_work[b] : X_base[b];
      const int p = static_cast<int>(Xb.n_cols);

      arma::vec wv = Rcpp::as<arma::vec>(Wk[b]);
      if (static_cast<int>(wv.n_elem) != p) {
        Rcpp::stop(std::string("cpp_compute_test_ev_core: weight length mismatch at component ") +
                   std::to_string(k + 1) + ", block " + std::to_string(b + 1) +
                   ". Expected " + std::to_string(p) + " entries but received " + std::to_string(wv.n_elem) + ".");
      }

      arma::vec tb = Xb * wv;
      const double v = arma::var(tb);
      if (std::isfinite(v) && v > eps_var) {
        Tk.col(b) = tb;
        valid_block(k, b) = 1;
      }
    }

    T_mat.cols(k * B, (k + 1) * B - 1) = Tk;

    double acc = 0.0;
    int n_pairs = 0;
    if (B >= 2) {
      for (int i = 0; i < B - 1; ++i) {
        for (int j = i + 1; j < B; ++j) {
          if (!(valid_block(k, i) && valid_block(k, j))) continue;
          const double r = compute_correlation_core(Tk.col(i), Tk.col(j), spearman);
          if (std::isfinite(r)) {
            acc += frobenius ? (r * r) : std::abs(r);
            ++n_pairs;
          }
        }
      }
    }
    // n_pairs == 0 means no identifiable cross-block pair: fewer than two blocks
    // have non-degenerate (non-zero variance) scores. NaN records that the
    // correlation is undefined here, rather than an observed zero; the R measures
    // score such a component as zero association.
    mac_comp(k) = (n_pairs > 0) ? (frobenius ? std::sqrt(acc) : acc / n_pairs) : arma::datum::nan;

    double ss_exp_total_k = 0.0;
    for (int b = 0; b < B; ++b) {
      const arma::mat& Xb_cur = deflate ? X_work[b] : X_base[b];
      const int p = static_cast<int>(Xb_cur.n_cols);
      arma::vec tb = Tk.col(b);

      arma::vec pb(p, arma::fill::zeros);
      if (use_train_loadings && P_all.size() > k) {
        arma::vec p_in = Rcpp::as<arma::vec>(Pk[b]);
        if (static_cast<int>(p_in.n_elem) != p) {
          Rcpp::stop(std::string("cpp_compute_test_ev_core: loading length mismatch at component ") +
                     std::to_string(k + 1) + ", block " + std::to_string(b + 1) +
                     ". Expected " + std::to_string(p) + " entries but received " + std::to_string(p_in.n_elem) + ".");
        }
        pb = p_in;
      } else {
        const double denom = arma::dot(tb, tb);
        if (std::isfinite(denom) && denom > 1e-12) {
          pb = (Xb_cur.t() * tb) / denom;
        }
      }

      const double ss_before = arma::accu(Xb_cur % Xb_cur);
      arma::mat X_new = Xb_cur - tb * pb.t();
      const double ss_after = arma::accu(X_new % X_new);

      const double ss_exp_block_raw = ss_before - ss_after;
      ss_exp_total_k += ss_exp_block_raw;

      const double inc_ratio = (ss_tot_test(b) > 1e-12) ? (ss_exp_block_raw / ss_tot_test(b)) : 0.0;
      ev_block_inc(k, b) = clamp_ratio(inc_ratio);

      ss_exp_cum_block(b) += ss_exp_block_raw;
      const double cum_ratio = (ss_tot_test(b) > 1e-12) ? (ss_exp_cum_block(b) / ss_tot_test(b)) : 0.0;
      ev_block_cum(k, b) = clamp_ratio(cum_ratio);

      if (deflate) {
        X_work[b] = std::move(X_new);
      }
    }

    const double inc_comp = (ss_tot_all > 1e-12) ? (ss_exp_total_k / ss_tot_all) : 0.0;
    ev_comp_inc(k) = clamp_ratio(inc_comp);

    ss_exp_cum_total += ss_exp_total_k;
    const double cum_comp = (ss_tot_all > 1e-12) ? (ss_exp_cum_total / ss_tot_all) : 0.0;
    ev_comp_cum(k) = clamp_ratio(cum_comp);
  }

  Rcpp::LogicalMatrix valid_out(K, B);
  for (int k = 0; k < K; ++k) {
    for (int b = 0; b < B; ++b) {
      valid_out(k, b) = static_cast<bool>(valid_block(k, b));
    }
  }

  return Rcpp::List::create(
    Rcpp::_["ev_block"] = ev_block_inc,
    Rcpp::_["ev_comp"] = ev_comp_inc,
    Rcpp::_["ev_block_cum"] = ev_block_cum,
    Rcpp::_["ev_comp_cum"] = ev_comp_cum,
    Rcpp::_["mac_comp"] = mac_comp,
    Rcpp::_["valid_block"] = valid_out,
    Rcpp::_["T_mat"] = T_mat
  );
}

// ─────────────────────────────────────────────────────────────────────
//  Helper: align shapes and filter invalid blocks
//    • trims each X_b / W_b to the common min width
//    • drops non‑finite / near‑zero blocks
//    • returns only the valid, aligned blocks
// ─────────────────────────────────────────────────────────────────────
static inline
std::pair<std::vector<arma::mat>, std::vector<arma::vec>>
align_and_filter(const std::vector<arma::mat>& Xin,
                 const std::vector<arma::vec>& Win)
{
  if (Xin.size() != Win.size()) {
    Rcpp::stop("Prediction-side validation requires one weight vector per data block.");
  }
  std::vector<arma::mat> Xv;
  std::vector<arma::vec> Wv;
  Xv.reserve(Xin.size());
  Wv.reserve(Win.size());

  arma::uword expected_rows = Xin.empty() ? 0 : Xin[0].n_rows;
  for (size_t b = 0; b < Xin.size(); ++b) {
    arma::mat Xb = Xin[b];
    arma::vec Wb = Win[b];

    if (!Xb.is_finite() || !Wb.is_finite()) {
      Rcpp::stop(std::string("Prediction-side validation received non-finite data in block ") + std::to_string(b + 1) + ".");
    }
    if (Xb.n_rows == 0 || Xb.n_cols == 0 || Wb.n_elem == 0) {
      Rcpp::stop(std::string("Prediction-side validation received an empty block or weight vector in block ") + std::to_string(b + 1) + ".");
    }
    if (Xb.n_rows != expected_rows) {
      Rcpp::stop(std::string("Prediction-side validation row mismatch in block ") +
                 std::to_string(b + 1) + ": expected " +
                 std::to_string(expected_rows) + " rows but found " +
                 std::to_string(Xb.n_rows) + ".");
    }

    const int pX = static_cast<int>(Xb.n_cols);
    const int pW = static_cast<int>(Wb.n_elem);
    if (pX != pW) {
      Rcpp::stop(std::string("Prediction-side validation feature/weight mismatch in block ") + std::to_string(b + 1) +
                 ": block has " + std::to_string(pX) + " columns but the weight vector has " + std::to_string(pW) + " entries.");
    }

    Xv.push_back(std::move(Xb));
    Wv.push_back(std::move(Wb));
  }
  return std::make_pair(std::move(Xv), std::move(Wv));
}


// [[Rcpp::export]]
Rcpp::List cpp_perm_test_oos(
    const Rcpp::List& X_test,             // list of test blocks (n x p_b)
    const Rcpp::List& W_trained,          // list of trained weight vectors (p_b)
    int               n_perm               = 1000,
    bool              spearman             = false,
    bool              frobenius            = false,
    double            early_stop_threshold = 1.0,   // retained for API compatibility; ignored
    bool              permute_all_blocks   = true)  // B==2: may set false to permute only block 2
{
  const int B = X_test.size();
  if (B < 2) Rcpp::stop("Need at least 2 blocks");
  if (W_trained.size() != B) {
    Rcpp::stop("cpp_perm_test_oos requires one trained weight vector per test block.");
  }
  if (n_perm < 1) {
    Rcpp::stop("cpp_perm_test_oos requires at least one sampled permutation.");
  }
  (void)early_stop_threshold;  // Retained for API compatibility; full B is used.

  // Materialize X and W from R lists
  std::vector<arma::mat> X(B);
  std::vector<arma::vec> W(B);

  int n = -1;
  for (int b = 0; b < B; ++b) {
    X[b] = Rcpp::as<arma::mat>(X_test[b]);
    W[b] = Rcpp::as<arma::vec>(W_trained[b]);

    if (!is_valid_matrix(X[b]) || !is_valid_vector(W[b])) {
      Rcpp::stop(std::string("cpp_perm_test_oos: invalid matrix or weight vector in block ") + std::to_string(b + 1) + ".");
    }
    if (n == -1) n = static_cast<int>(X[b].n_rows);
  }

  if (n < 3) {
    Rcpp::stop("cpp_perm_test_oos requires at least 3 rows in every block.");
  }

  auto pr_obs = align_and_filter(X, W);
  const auto& Xv_obs = pr_obs.first;
  const auto& Wv_obs = pr_obs.second;
  const ScoreMatrix obs_scores = compute_scores_core(Xv_obs, Wv_obs);
  if (obs_scores.n_valid < 2) {
    Rcpp::stop("cpp_perm_test_oos requires at least two blocks with non-degenerate score vectors in the observed data.");
  }

  const double stat_obs = compute_objective_direct_core(Xv_obs, Wv_obs, spearman, frobenius);

  // Prepare row indices for permutations
  std::vector<arma::uvec> base_idx(B);
  for (int b = 0; b < B; ++b)
    base_idx[b] = arma::regspace<arma::uvec>(0, n - 1);

  int ge = 0;

  for (int p = 0; p < n_perm; ++p) {
    // Permute rows (all blocks or all but the first)
    std::vector<arma::mat> Xp = X;
    if (permute_all_blocks) {
      for (int b = 0; b < B; ++b) {
        if (Xp[b].n_rows == static_cast<arma::uword>(n)) {
          Xp[b] = X[b].rows(arma::shuffle(base_idx[b]));
        }
      }
    } else {
      for (int b = 1; b < B; ++b) {
        if (Xp[b].n_rows == static_cast<arma::uword>(n)) {
          Xp[b] = X[b].rows(arma::shuffle(base_idx[b]));
        }
      }
    }

    auto pr_perm = align_and_filter(Xp, W);
    const auto& Xv_perm = pr_perm.first;
    const auto& Wv_perm = pr_perm.second;
    const ScoreMatrix perm_scores = compute_scores_core(Xv_perm, Wv_perm);
    if (perm_scores.n_valid < 2) {
      Rcpp::stop(std::string("cpp_perm_test_oos: permutation replicate ") + std::to_string(p + 1) +
                 " produced fewer than two valid score blocks.");
    }
    const double stat_perm = compute_objective_direct_core(Xv_perm, Wv_perm, spearman, frobenius);
    if (stat_perm >= stat_obs - 100.0 * std::numeric_limits<double>::epsilon() * std::abs(stat_obs)) ++ge;

  }

  const double pval = (ge + 1.0) / (n_perm + 1.0);  // add-one smoothing
  return Rcpp::List::create(
    Rcpp::_["stat_obs"] = stat_obs,
    Rcpp::_["p_value"]  = pval,
    Rcpp::_["n_perm"]   = n_perm,
    Rcpp::_["n_extreme"] = ge,
    Rcpp::_["alternative"] = "greater",
    Rcpp::_["correction"] = "(b + 1) / (B + 1)"
  );
}


// [[Rcpp::export]]
Rcpp::List cpp_bootstrap_test_oos(
    const Rcpp::List& X_test,
    const Rcpp::List& W_trained,
    int               n_boot    = 1000,
    bool              spearman  = false,
    bool              frobenius = false,
    double            alpha     = 0.05)
{
  const int B = X_test.size();
  if (B < 2) Rcpp::stop("Need at least 2 blocks");
  if (W_trained.size() != B) {
    Rcpp::stop("cpp_bootstrap_test_oos requires one trained weight vector per test block.");
  }
  if (n_boot < 2) {
    Rcpp::stop("cpp_bootstrap_test_oos requires at least two bootstrap replicates.");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
    Rcpp::stop("cpp_bootstrap_test_oos requires alpha strictly between 0 and 1.");
  }

  std::vector<arma::mat> X(B);
  std::vector<arma::vec> W(B);

  int n = -1;
  for (int b = 0; b < B; ++b) {
    X[b] = Rcpp::as<arma::mat>(X_test[b]);
    W[b] = Rcpp::as<arma::vec>(W_trained[b]);

    if (!is_valid_matrix(X[b]) || !is_valid_vector(W[b])) {
      Rcpp::stop(std::string("cpp_bootstrap_test_oos: invalid matrix or weight vector in block ") + std::to_string(b + 1) + ".");
    }
    if (n == -1) n = static_cast<int>(X[b].n_rows);
  }

  if (n < 3) {
    Rcpp::stop("cpp_bootstrap_test_oos requires at least 3 rows in every block.");
  }

  auto pr_obs = align_and_filter(X, W);
  const auto& Xv_obs = pr_obs.first;
  const auto& Wv_obs = pr_obs.second;

  const int Bv = static_cast<int>(Xv_obs.size());
  const int n_aligned = static_cast<int>(Xv_obs[0].n_rows);
  if (n_aligned < 3) {
    Rcpp::stop("cpp_bootstrap_test_oos requires at least 3 aligned rows in the observed data.");
  }

  std::vector<arma::vec> T_full;
  T_full.reserve(Bv);
  for (int b = 0; b < Bv; ++b) {
    arma::vec tb = Xv_obs[b] * Wv_obs[b];
    T_full.push_back(std::move(tb));
  }

  std::vector<int> valid_blocks;
  valid_blocks.reserve(Bv);
  for (int b = 0; b < Bv; ++b) {
    if (is_valid_vector(T_full[b])) {
      double v = arma::var(T_full[b]);
      if (std::isfinite(v) && v > 1e-12) {
        valid_blocks.push_back(b);
      }
    }
  }

  if (static_cast<int>(valid_blocks.size()) < 2) {
    Rcpp::stop("cpp_bootstrap_test_oos requires at least two blocks with non-degenerate score vectors in the observed data.");
  }

  auto stat_from_scores = [&](const std::vector<arma::vec>& Tset, const arma::uvec* idx) {
    double acc = 0.0;
    int pairs = 0;
    for (size_t ii = 0; ii < valid_blocks.size() - 1; ++ii) {
      for (size_t jj = ii + 1; jj < valid_blocks.size(); ++jj) {
        const int bi = valid_blocks[ii];
        const int bj = valid_blocks[jj];
        arma::vec xi = (idx == nullptr) ? Tset[bi] : Tset[bi].elem(*idx);
        arma::vec xj = (idx == nullptr) ? Tset[bj] : Tset[bj].elem(*idx);
        double r = compute_correlation_core(xi, xj, spearman);
        if (!std::isfinite(r)) {
          // A replicate must retain the observed set of block pairs. Dropping
          // degenerate pairs would bootstrap a different association statistic.
          return NA_REAL;
        }
        acc += frobenius ? (r * r) : std::abs(r);
        ++pairs;
      }
    }
    if (pairs == 0) {
      Rcpp::stop("cpp_bootstrap_test_oos: no valid score pairs remain for the requested statistic.");
    }
    return frobenius ? std::sqrt(acc) : (acc / pairs);
  };

  const double stat_obs = stat_from_scores(T_full, nullptr);

  arma::vec boot_vals(n_boot, arma::fill::zeros);
  int valid_reps = 0;

  for (int r = 0; r < n_boot; ++r) {
    arma::uvec idx = arma::randi<arma::uvec>(n_aligned, arma::distr_param(0, n_aligned - 1));
    try {
      double stat_boot = stat_from_scores(T_full, &idx);
      if (std::isfinite(stat_boot)) {
        boot_vals(valid_reps) = stat_boot;
        ++valid_reps;
      }
    } catch (const std::exception&) {
      // A degenerate resample is recorded as failed and excluded from summaries.
    }
  }

  if (valid_reps < 2) {
    Rcpp::stop("cpp_bootstrap_test_oos: fewer than two bootstrap replicates produced valid statistics.");
  }

  arma::vec vals = boot_vals.head(valid_reps);
  const double boot_mean = arma::mean(vals);
  const double boot_se = arma::stddev(vals);

  arma::vec sorted_vals = arma::sort(vals);
  auto quantile_type8 = [&](double probability) {
    const double n_vals = static_cast<double>(sorted_vals.n_elem);
    const double h = (n_vals + 1.0 / 3.0) * probability + 1.0 / 3.0;
    const int j = static_cast<int>(std::floor(h));
    const double gamma = h - static_cast<double>(j);
    if (j <= 0) return sorted_vals(0);
    if (j >= static_cast<int>(sorted_vals.n_elem)) {
      return sorted_vals(sorted_vals.n_elem - 1);
    }
    return (1.0 - gamma) * sorted_vals(j - 1) + gamma * sorted_vals(j);
  };
  const double ci_lower = quantile_type8(alpha / 2.0);
  const double ci_upper = quantile_type8(1.0 - alpha / 2.0);

  return Rcpp::List::create(
    Rcpp::_["stat_obs"]  = stat_obs,
    Rcpp::_["boot_mean"] = boot_mean,
    Rcpp::_["bias"]      = boot_mean - stat_obs,
    Rcpp::_["boot_se"]   = boot_se,
    Rcpp::_["p_value"]   = NA_REAL,
    Rcpp::_["p_value_note"] = "Not computed: an ordinary bootstrap distribution is not a null distribution for hypothesis testing.",
    Rcpp::_["ci_lower"]  = ci_lower,
    Rcpp::_["ci_upper"]  = ci_upper,
    Rcpp::_["confidence_level"] = 1.0 - alpha,
    Rcpp::_["interval_type"] = "percentile",
    Rcpp::_["n_boot"]    = valid_reps,
    Rcpp::_["n_boot_requested"] = n_boot,
    Rcpp::_["n_boot_failed"] = n_boot - valid_reps,
    Rcpp::_["replicates"] = vals
  );
}
