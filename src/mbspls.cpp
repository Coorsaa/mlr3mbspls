// =====================================================================
//  File: src/mbspls.cpp
//  -------------------------------------------------------------------
//  C++ / Armadillo implementation of unsupervised multi-block sparse PLS
//  (MB-sPLS) with permutation-test based early stopping.
//  Functions exported to R:
//    • cpp_mbspls_one_lv()              - one-component solver
//    • cpp_mbspls_multi_lv()            - multi-component solver
// =====================================================================
#ifndef MBSPLS_L2_BETA
#define MBSPLS_L2_BETA 0.5   // β_b default; 0.5 makes w = soft(g, α) exactly
#endif

#define ARMA_DONT_ALIGN_MEMORY
#include <RcppArmadillo.h>
#include <utility>  // std::pair

using arma::uvec;
using arma::vec;
using arma::mat;
using arma::cube;
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

inline arma::vec safe_normalize(const arma::vec& v) {
  if (!is_valid_vector(v)) {
    Rcpp::stop("safe_normalize: received a non-finite or empty vector.");
  }

  double norm_val = arma::norm(v, 2);
  if (!std::isfinite(norm_val) || norm_val < 1e-12) {
    Rcpp::stop("safe_normalize: vector norm is numerically zero; cannot normalize weights.");
  }

  return v / norm_val;
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


// ---- Weight normalization controls ---------------------------------
#ifndef MBSPLS_WEIGHT_NORM
// 0: none, 1: L1 (sum|w| = 1), 2: L2 (||w||2 = 1), 3: maxabs (max|w| = 1)
#define MBSPLS_WEIGHT_NORM 2
#endif

#ifndef MBSPLS_NORM_EPS
#define MBSPLS_NORM_EPS 1e-12
#endif

inline void mbspls_apply_weight_norm(arma::vec &w) {
#if MBSPLS_WEIGHT_NORM==1
  double l1 = arma::accu(arma::abs(w));
  if (std::isfinite(l1) && l1 > MBSPLS_NORM_EPS) w /= l1;
#elif MBSPLS_WEIGHT_NORM==2
  double l2 = arma::norm(w, 2);
  if (std::isfinite(l2) && l2 > MBSPLS_NORM_EPS) w /= l2;
#elif MBSPLS_WEIGHT_NORM==3
  double m = (w.is_empty() ? 0.0 : arma::abs(w).max());
  if (std::isfinite(m) && m > MBSPLS_NORM_EPS) w /= m;
#else
  (void)w; // no-op
#endif
}

// ─────────────────────────────────────────────────────────────────────
//  SOFT-THRESHOLDING UTILITIES
// ─────────────────────────────────────────────────────────────────────

// --- helper: plain soft threshold (no extra scaling) ---
inline arma::vec soft_no_scale(const arma::vec& g, double alpha) {
  return arma::sign(g) % arma::max(arma::abs(g) - alpha,
                                   arma::zeros<arma::vec>(g.n_elem));
}

inline arma::vec pmd_update_bisection(const arma::vec& g,
                                      double c,
                                      int    maxit = 60,
                                      double tol   = 1e-8)
{
  const arma::uword p = g.n_elem;
  if (p == 0) {
    Rcpp::stop("pmd_update_bisection: empty gradient vector.");
  }

  if (!g.is_finite() || arma::norm(g, 2) < 1e-16) {
    Rcpp::stop("pmd_update_bisection: gradient is non-finite or numerically zero; cannot compute a sparse weight vector.");
  }

  if (!std::isfinite(c) || c >= std::sqrt(static_cast<double>(p))) {
    arma::vec w = g;
    double n2 = arma::norm(w, 2);
    if (!std::isfinite(n2) || n2 < 1e-12) {
      Rcpp::stop("pmd_update_bisection: unconstrained normalization failed because the gradient norm is numerically zero.");
    }
    return w / n2;
  }

  double lo = 0.0, hi = arma::abs(g).max();
  if (!std::isfinite(hi) || hi <= 0.0) {
    Rcpp::stop("pmd_update_bisection: bisection bracket is degenerate because the gradient has no finite magnitude.");
  }

  arma::vec w; double l1 = 0.0;

  for (int it = 0; it < maxit; ++it) {
    const double alpha = 0.5 * (lo + hi);
    arma::vec z = soft_no_scale(g, alpha);
    double n2 = arma::norm(z, 2);

    if (n2 < 1e-16) { hi = alpha; continue; }    // too much shrinkage

    w  = z / n2;                                  // enforce ||w||2 = 1
    l1 = arma::accu(arma::abs(w));
    if (std::abs(l1 - c) <= tol) break;

    if (l1 > c) lo = alpha; else hi = alpha;      // move the bracket
  }

  // Final tighten if slightly above c
  if (l1 > c + 1e-10) {
    arma::vec z = soft_no_scale(g, hi);
    double n2 = arma::norm(z, 2);
    if (n2 >= 1e-16) w = z / n2;
  }
  if (w.n_elem != p || !w.is_finite()) {
    Rcpp::stop("pmd_update_bisection: produced a non-finite weight vector.");
  }
  const double w_norm = arma::norm(w, 2);
  if (!std::isfinite(w_norm) || w_norm < 1e-12) {
    Rcpp::stop("pmd_update_bisection: produced a numerically zero weight vector.");
  }
  return w;
}


// ─────────────────────────────────────────────────────────────────────
//  EXPORTED FUNCTIONS - R Interface (using core functions internally)
// ─────────────────────────────────────────────────────────────────────

// [[Rcpp::export]]
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
  if (c_constraints.n_elem != B) Rcpp::stop("c_constraints length must match number of blocks");

  std::vector<arma::mat> X;
  std::vector<arma::mat> Xt;
  X.reserve(B);
  Xt.reserve(B);
  int n = -1;
  
  for (int b = 0; b < B; ++b) {
    Rcpp::NumericMatrix Xr = X_blocks[b];
    arma::mat Xi(Xr.begin(), Xr.nrow(), Xr.ncol(), false, true);
    if (!is_valid_matrix(Xi)) {
      Rcpp::stop("Invalid matrix in block " + std::to_string(b + 1));
    }
    
    if (n == -1) {
      n = Xi.n_rows;
    } else if (Xi.n_rows != n) {
      Rcpp::stop("Inconsistent sample sizes across blocks");
    }
    
    if (n < 3) Rcpp::stop("Need at least 3 samples");
    X.emplace_back(Xr.begin(), Xr.nrow(), Xr.ncol(), false, true);
    Xt.emplace_back(X.back().t());
  }

  // Initialize weights
  std::vector<arma::vec> W(B);
  for (int b = 0; b < B; ++b) {
    W[b] = arma::randn<arma::vec>(X[b].n_cols);
    W[b] = safe_normalize(W[b]);
  }

  double obj_old = -1e6;
  bool converged = false;

  for (int it = 0; it < max_iter; ++it) {
    
    // CORE: Use standardized score computation
    ScoreMatrix current_scores = compute_scores_core(X, W);

    for (int b = 0; b < B; ++b) {
      // CORE: Use standardized target building
      arma::vec target = build_target_score_core(current_scores, b);
      
      if (arma::norm(target, 2) < 1e-12) {
        Rcpp::stop(std::string("cpp_mbspls_one_lv: no valid cross-block target score could be formed for block ") + std::to_string(b + 1) + ". Check that at least two blocks contain informative, non-degenerate signals.");
      }

      arma::vec grad = Xt[b] * target;

      if (!is_valid_vector(grad)) {
        Rcpp::stop(std::string("cpp_mbspls_one_lv: gradient became non-finite for block ") + std::to_string(b + 1) + ".");
      }

      // Enforce L2 = 1 and L1 ≈ c via bisection (PMD update)
      W[b] = pmd_update_bisection(grad, c_constraints(b));

    }

    // CORE: Use standardized objective computation
    double obj = compute_objective_direct_core(X, W, spearman, frobenius);

    if (std::abs(obj - obj_old) < tol) {
      log_info("     one-LV solver converged after " + std::to_string(it+1) + " iterations");
      converged = true;
      break;
    }
    obj_old = obj;
  }

  // Build final results using core computation
  ScoreMatrix final_scores = compute_scores_core(X, W);

  Rcpp::List W_out(B); 
  for (int b = 0; b < B; ++b) {
    W_out[b] = W[b];
  }

  return Rcpp::List::create(
    Rcpp::_["W"]         = W_out,
    Rcpp::_["T_mat"]     = final_scores.T,
    Rcpp::_["objective"] = obj_old,
    Rcpp::_["converged"] = converged
  );
}

// // Procrustes function (from original)
// // [[Rcpp::export(rng = false)]]
// arma::mat orth_procrustes(const arma::mat &A, const arma::mat &B) {
//   if (A.n_rows != B.n_rows || A.n_cols != B.n_cols)
//     Rcpp::stop("A and B must have identical shape for Procrustes rotation");

//   arma::mat U, V; arma::vec s;
//   arma::svd_econ(U, s, V, A.t() * B);

//   arma::mat R = V * U.t();
//   if (arma::det(R) < 0) {
//     V.col(V.n_cols - 1) *= -1.0;
//     R = V * U.t();
//   }
//   return R;
// }

// // [[Rcpp::export]]
// double perm_test_component(
//     const std::vector<arma::mat> &X_orig,
//     const std::vector<arma::vec> &W_orig,
//     const arma::vec             &c_vec,
//     int                           n_perm   = 1000,
//     bool                          spearman = false,
//     int                           max_iter = 500,
//     double                        tol      = 1e-4,
//     double                        early_stop_threshold = 0.05,
//     bool                          frobenius = false
//   )
// {
//   const int B = static_cast<int>(X_orig.size());
//   const int n = static_cast<int>(X_orig[0].n_rows);

//   // Reference: scores & objective on the original, aligned blocks
//   ScoreMatrix ref_scores = compute_scores_core(X_orig, W_orig);
//   arma::mat T_ref = ref_scores.T;
//   T_ref.each_row() -= arma::mean(T_ref, 0);
//   const double obj_ref = compute_block_objective_core(ref_scores, spearman, frobenius);

//   int ge = 0;

//   for (int p = 0; p < n_perm; ++p) {
//     // ────────────────────────────────────────────────────────────────
//     // Permute ALL blocks independently to break cross-block alignment
//     // ────────────────────────────────────────────────────────────────
//     std::vector<arma::mat> X = X_orig;
//     for (int b = 0; b < B; ++b) {
//       arma::uvec idx = arma::shuffle(arma::regspace<arma::uvec>(0, n - 1));
//       X[b] = X[b].rows(idx);
//     }

//     // Optional early-stop on running p-value
//     if (p > 100 && p % 50 == 0) {
//       double current_p = static_cast<double>(ge + 1) / (p + 1);
//       if (current_p > early_stop_threshold) {
//         return current_p;
//       }
//     }

//     // Fit one-LV to the permuted blocks
//     Rcpp::List X_list(B);
//     for (int b = 0; b < B; ++b) X_list[b] = X[b];

//     try {
//       Rcpp::List fit = cpp_mbspls_one_lv(X_list, c_vec, max_iter, tol, frobenius);

//       // Unwrap weights
//       std::vector<arma::vec> Wp(B);
//       Rcpp::List Wtmp = fit["W"];
//       for (int b = 0; b < B; ++b) Wp[b] = Rcpp::as<arma::vec>(Wtmp[b]);

//       // Compute & mean-center permuted scores
//       ScoreMatrix perm_scores = compute_scores_core(X, Wp);
//       arma::mat T_perm = perm_scores.T;
//       T_perm.each_row() -= arma::mean(T_perm, 0);

//       // Procrustes align to reference scores to remove sign/rotation ambiguity
//       arma::mat R = orth_procrustes(T_ref, T_perm);
//       T_perm = T_perm * R;

//       // Evaluate objective on aligned permuted scores
//       ScoreMatrix aligned_scores(n, B);
//       aligned_scores.T = T_perm;
//       aligned_scores.valid_blocks = perm_scores.valid_blocks;
//       aligned_scores.n_valid = perm_scores.n_valid;

//       double obj_perm = compute_block_objective_core(aligned_scores, spearman, frobenius);
//       if (obj_perm >= obj_ref) ++ge;

//     } catch (...) {
//       // If a replicate fails to fit, skip it
//       continue;
//     }
//   }

//   return (ge + 1.0) / (n_perm + 1.0);
// }

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
  const int n = static_cast<int>(X_orig[0].n_rows);

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

    // Optional early stop on running p-value
    if (early_stop_threshold < 1.0 && p >= 100 && (p % 50) == 0) {
      // Safe lower bound on the final p-value if we stopped now
      double p_lower_final = static_cast<double>(ge + 1) / static_cast<double>(n_perm + 1);
      if (p_lower_final > early_stop_threshold) {
        return p_lower_final;
      }
    }

    // Fit one-LV to the permuted blocks using the SAME penalties
    Rcpp::List X_list(B);
    for (int b = 0; b < B; ++b) X_list[b] = X[b];

    try {
      Rcpp::List fit = cpp_mbspls_one_lv(X_list, c_vec, max_iter, tol, frobenius);

      // Unwrap weights
      std::vector<arma::vec> Wp(B);
      Rcpp::List Wtmp = fit["W"];
      for (int b = 0; b < B; ++b) Wp[b] = Rcpp::as<arma::vec>(Wtmp[b]);

      // Evaluate the SAME statistic on permuted fit — NO rotations/Procrustes
      const double obj_perm = compute_objective_direct_core(X, Wp, spearman, frobenius);

      if (obj_perm >= obj_ref) ++ge;

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

    Rcpp::List X_list(B); 
    for (int b = 0; b < B; ++b) X_list[b] = X[b];
    
    Rcpp::List fit;
    try {
      fit = cpp_mbspls_one_lv(X_list, c_constraints, max_iter, tol, frobenius);
    } catch (const std::exception &e) {
      Rcpp::stop(std::string("Component extraction failed at component ") + std::to_string(k + 1) + ": " + e.what());
    }

    std::vector<arma::vec> Wk(B);
    Rcpp::List Wtmp = fit["W"];
    for (int b = 0; b < B; ++b) Wk[b] = Rcpp::as<arma::vec>(Wtmp[b]);

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
    T_all = arma::join_rows(T_all, Tk);

    std::vector<arma::vec> Pk(B);
    arma::vec ev_block(B);
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
      
      if (!deflate_block(X[b], tb, pb)) {
        Rcpp::stop(std::string("cpp_mbspls_multi_lv: deflation failed for block ") +
                   std::to_string(b + 1) + ", component " + std::to_string(k + 1) + ".");
      }
    }

    double ev_comp = ss_exp_total / arma::accu(ss_tot);

    bool keep_component = (!do_perm) || keep_it || k == 0;   // always keep LV-1

    if (!keep_component) {           // non-significant and NOT LV-1
      log_info("🚦  LV not significant - stopping extraction");
      break;                         // nothing stored yet
    }

    /* ---------- store the component ---------- */
    W_all.push_back(Wk);
    P_all.push_back(Pk);
    obj_vec.push_back(obj_k);
    p_vec.push_back(p_val);
    ev_block_list.push_back(ev_block);
    ev_comp_list.push_back(ev_comp);

    /* ---------- stop right after LV-1 if it was not significant ---------- */
    if (do_perm && !keep_it) {       // this can only be k == 0
      log_info("🚦  LV-1 kept but not significant - no further extraction");
      break;
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
    Rcpp::_["ev_comp"]    = ev_comp_vec
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

  for (int k = 0; k < K; ++k) {
    log_info("⏩  extracting component " + std::to_string(k + 1));

    // Current sparsity vector
    arma::vec c_vec = c_matrix.col(k);
    if (static_cast<int>(c_vec.n_elem) != B)
      Rcpp::stop("Unexpected c_matrix column length at component %d", k + 1);

    // Fit one LV on the *current* (already-deflated up to k-1) blocks
    Rcpp::List X_list(B);
    for (int b = 0; b < B; ++b) X_list[b] = X[b];

    Rcpp::List fit;
    try {
      // IMPORTANT: pass spearman through
      fit = cpp_mbspls_one_lv(X_list, c_vec, max_iter, tol, frobenius, spearman);
    } catch (const std::exception &e) {
      Rcpp::stop(std::string("Component extraction failed at component ") + std::to_string(k + 1) + ": " + e.what());
    }

    // Unwrap weights
    std::vector<arma::vec> Wk(B);
    {
      Rcpp::List Wtmp = fit["W"];
      for (int b = 0; b < B; ++b) Wk[b] = Rcpp::as<arma::vec>(Wtmp[b]);
    }

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
    T_all = arma::join_rows(T_all, Tk);

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

      // Deflate for next component
      if (!deflate_block(X[b], tb, pb)) {
        Rcpp::stop(std::string("cpp_mbspls_multi_lv_cmatrix: deflation failed for block ") +
                   std::to_string(b + 1) + ", component " + std::to_string(k + 1) + ".");
      }
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
    W_all.push_back(Wk);
    P_all.push_back(Pk);
    obj_vec.push_back(obj_k);
    p_vec.push_back(p_val);
    ev_block_list.push_back(ev_block);
    ev_comp_list.push_back(ev_comp);

    // If LV-1 was not significant: keep it but stop afterwards
    if (do_perm && !keep_it) {
      log_info("🚦  LV-1 kept but not significant - no further extraction");
      break;
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
    Rcpp::_["ev_comp"]    = ev_comp_vec
  );
}


// [[Rcpp::export]]
Rcpp::List cpp_ev_test(const Rcpp::List&  X_test,
                       const Rcpp::List&  weights,
                       const Rcpp::List&  loadings,
                       int                ncomp)
{
  const int B = X_test.size();
  
  if (B == 0 || ncomp <= 0) {
    return Rcpp::List::create(
      Rcpp::_["block"] = arma::vec(B, arma::fill::zeros),
      Rcpp::_["total"] = 0.0);
  }
  
  arma::vec ss_tot(B, arma::fill::zeros), ss_exp(B, arma::fill::zeros);

  std::vector<arma::mat> X(B);
  for (int b = 0; b < B; ++b) {
    X[b] = Rcpp::as<arma::mat>(X_test[b]);
    ss_tot(b) = arma::accu(arma::square(X[b]));
  }

  // Loop over components
  for (int k = 0; k < ncomp; ++k) {
    // Unwrap weights / loadings for component k
    Rcpp::List Wk_l = weights[k], Pk_l = loadings[k];
    std::vector<arma::vec> Wk(B), Pk(B);
    for (int b = 0; b < B; ++b) {
      Wk[b] = Rcpp::as<arma::vec>(Wk_l[b]);
      Pk[b] = Rcpp::as<arma::vec>(Pk_l[b]);
    }

    // Project + deflate
    for (int b = 0; b < B; ++b) {
      if (Wk[b].is_empty()) continue;
      arma::vec tb = X[b] * Wk[b];
      double norm2 = arma::dot(tb, tb);
      if (norm2 < 1e-12) continue;

      ss_exp(b) += norm2 * arma::dot(Pk[b], Pk[b]);
      X[b] -= tb * Pk[b].t();  // Deflate for next component
    }
  }

  arma::vec ev_block = arma::vec(B, arma::fill::zeros);
  double ss_all = arma::accu(ss_tot);
  
  // Calculate EV with simple finite checks
  for (int b = 0; b < B; ++b) {
    if (ss_tot(b) > 1e-12) {
      double ratio = ss_exp(b) / ss_tot(b);
      ev_block(b) = std::isfinite(ratio) ? std::max(0.0, std::min(1.0, ratio)) : 0.0;
    }
  }

  double ev_total = 0.0;
  if (ss_all > 1e-12) {
    double ratio = arma::accu(ss_exp) / ss_all;
    ev_total = std::isfinite(ratio) ? std::max(0.0, std::min(1.0, ratio)) : 0.0;
  }

  return Rcpp::List::create(
    Rcpp::_["block"] = ev_block,
    Rcpp::_["total"] = ev_total);
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
    mac_comp(k) = (n_pairs > 0) ? (frobenius ? std::sqrt(acc) : acc / n_pairs) : 0.0;

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

// Bootstrap stability selection
// [[Rcpp::export]]
Rcpp::List cpp_mbspls_bootstrap(const Rcpp::List&  X_blocks,
                                const arma::vec&   c_constraints,
                                const Rcpp::List&  W_ref,
                                int                R             = 500,
                                bool               spearman      = false,
                                bool               frobenius     = false,
                                int                max_iter      = 500,
                                double             tol           = 1e-6,
                                bool               store_weights = true) {

  const int B = X_blocks.size();
  const int K = W_ref.size();
  if (B == 0 || K == 0)
    Rcpp::stop("empty input - need at least one block and one component");

  if (static_cast<int>(c_constraints.n_elem) != B)
    Rcpp::stop("c_constraints length must match number of blocks");

  // Basic sizes
  arma::mat X0 = Rcpp::as<arma::mat>(X_blocks[0]);
  const int n = X0.n_rows;
  if (n < 3)
    Rcpp::stop("need at least 3 samples");

  std::vector<int> block_sizes(B);
  int Ptot = 0;
  for (int b = 0; b < B; ++b) {
    arma::mat Xi = Rcpp::as<arma::mat>(X_blocks[b]);
    if (Xi.n_rows != n)
      Rcpp::stop("inconsistent sample size across blocks");
    block_sizes[b] = Xi.n_cols;
    Ptot += Xi.n_cols;
  }

  // Flatten reference weights
  std::vector<arma::vec> w_ref_flat(K);
  for (int k = 0; k < K; ++k) {
    Rcpp::List Wk = W_ref[k];
    if (static_cast<int>(Wk.size()) != B)
      Rcpp::stop("reference W wrong size");

    arma::vec flat(Ptot, arma::fill::zeros);
    int off = 0;
    for (int b = 0; b < B; ++b) {
      arma::vec wb = Rcpp::as<arma::vec>(Wk[b]);
      if (static_cast<int>(wb.n_elem) != block_sizes[b])
        Rcpp::stop("W size mismatch");
      flat.subvec(off, off + wb.n_elem - 1) = wb;
      off += wb.n_elem;
    }
    w_ref_flat[k] = flat;
  }

  // Containers
  arma::mat sel_freq(Ptot, K, arma::fill::zeros);
  arma::cube weight_store;
  if (store_weights)
    weight_store.set_size(Ptot, K, R);  // Dense cube only on request

  int ok_runs = 0;

  // Bootstrap loop
  for (int r = 0; r < R; ++r) {
    try {
      // 1. Resample indices (with replacement)
      arma::uvec idx = arma::randi<arma::uvec>(n, arma::distr_param(0, n - 1));

      // 2. Materialise resampled blocks
      Rcpp::List X_star(B);
      for (int b = 0; b < B; ++b) {
        arma::mat Xb = Rcpp::as<arma::mat>(X_blocks[b]);
        X_star[b] = arma::mat(Xb.rows(idx));
      }

      // 3. Fit MB-sPLS on resampled data
      Rcpp::List fit = cpp_mbspls_multi_lv(
        /* X_blocks */     X_star,
        /* c_constraints */c_constraints,
        /* K */            K,
        /* max_iter */     max_iter,
        /* tol */          tol,
        /* spearman */     spearman,
        /* do_perm */      false,
        /* n_perm */       100,
        /* alpha */        0.05,
        /* frobenius */    frobenius
      );

      Rcpp::List fitW = fit["W"];
      const int Kfit = fitW.size();

      // 4. Collect weights & selection freq.
      for (int k = 0; k < std::min(K, Kfit); ++k) {
        Rcpp::List Wk = fitW[k];

        arma::vec flat(Ptot, arma::fill::zeros);
        int off = 0;
        for (int b = 0; b < B; ++b) {
          arma::vec wb = Rcpp::as<arma::vec>(Wk[b]);
          flat.subvec(off, off + wb.n_elem - 1) = wb;
          off += wb.n_elem;
        }
        // Sign match
        if (arma::dot(flat, w_ref_flat[k]) < 0)
          flat = -flat;

        if (store_weights)
          weight_store.slice(ok_runs).col(k) = flat;

        sel_freq.col(k) += arma::conv_to<arma::vec>::from(
                             arma::abs(flat) > 1e-12);
      }

      ++ok_runs;

    } catch (const std::exception &e) {
      Rcpp::stop(std::string("Bootstrap replicate ") + std::to_string(r + 1) +
                 " failed while refitting MB-sPLS: " + e.what());
    }
  }

  // Post-processing
  if (ok_runs == 0)
    Rcpp::stop("All bootstrap replicates failed.");

  sel_freq /= ok_runs;

  if (store_weights)
    weight_store = weight_store.slices(0, ok_runs - 1);  // Shrink to filled

  // Build return list
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("freq")            = sel_freq,
    Rcpp::Named("successful_runs") = ok_runs
  );

  if (store_weights)
    out.push_back(Rcpp::wrap(weight_store), "weights");
  else
    out.push_back(R_NilValue, "weights");

  return out;
}

// Bootstrap latent correlation
// [[Rcpp::export]]
double cpp_bootstrap_latent_correlation(const arma::mat&  weights_matrix,
                                        const arma::ivec& component_idx,
                                        const arma::ivec& block_idx,
                                        int               n_blocks,
                                        int               n_components,
                                        bool              spearman   = false,
                                        double            min_var    = 1e-12,
                                        bool              frobenius  = false)    // ← NEW
{
  if (n_blocks < 2) {
    Rcpp::stop("cpp_bootstrap_latent_correlation: need at least two blocks to compute a latent correlation.");
  }

  double acc          = 0.0;        // collects  Σ|r|  or  Σr²
  int    n_comparisons = 0;
  
  // Process each component
  for (int k = 1; k <= n_components; ++k) {
    // Find features belonging to this component
    arma::uvec comp_mask = arma::find(component_idx == k);
    if (comp_mask.n_elem == 0) continue;
    
    // Group by blocks for this component
    std::vector<arma::vec> block_weights(n_blocks);
    std::vector<bool> block_has_data(n_blocks, false);
    
    for (arma::uword i = 0; i < comp_mask.n_elem; ++i) {
      arma::uword feat_idx = comp_mask(i);
      int block_id = block_idx(feat_idx) - 1; // Convert to 0-based
      
      if (block_id >= 0 && block_id < n_blocks) {
        if (!block_has_data[block_id]) {
          block_weights[block_id] = arma::vec();
          block_has_data[block_id] = true;
        }
        
        // Collect weights for this block (use absolute values)
        double weight_val = std::abs(weights_matrix(feat_idx, 0));
        block_weights[block_id] = arma::join_cols(block_weights[block_id], 
                                                  arma::vec{weight_val});
      }
    }
    
    // Count blocks with sufficient data
    std::vector<int> valid_blocks;
    for (int b = 0; b < n_blocks; ++b) {
      if (block_has_data[b] && block_weights[b].n_elem >= 3) {
        double var_b = arma::var(block_weights[b]);
        if (var_b > min_var) {
          valid_blocks.push_back(b);
        }
      }
    }
    
    // Compute pairwise correlations
    if (valid_blocks.size() >= 2) {
      for (size_t i = 0; i < valid_blocks.size() - 1; ++i) {
        for (size_t j = i + 1; j < valid_blocks.size(); ++j) {
          int b1 = valid_blocks[i];
          int b2 = valid_blocks[j];
          
          arma::vec w1 = block_weights[b1];
          arma::vec w2 = block_weights[b2];
          if (w1.n_elem != w2.n_elem) {
            Rcpp::stop(std::string("cpp_bootstrap_latent_correlation: block weight vectors have unequal lengths for component ") +
                       std::to_string(k) + ", blocks " + std::to_string(b1 + 1) + " and " + std::to_string(b2 + 1) +
                       ". Correlation is undefined without an explicit feature alignment.");
          }
          if (w1.n_elem >= 3) {
            double cor_val = 0.0;
            if (spearman) {
              // Convert to ranks for Spearman
              arma::vec r1 = arma::conv_to<arma::vec>::from(arma::sort_index(arma::sort_index(w1)));
              arma::vec r2 = arma::conv_to<arma::vec>::from(arma::sort_index(arma::sort_index(w2)));
              cor_val = arma::as_scalar(arma::cor(r1, r2));
            } else {
              cor_val = arma::as_scalar(arma::cor(w1, w2));
            }
            
            if (std::isfinite(cor_val)) {
              acc += frobenius ? cor_val * cor_val          // Σ r²
                               : std::abs(cor_val);         // Σ |r|
              ++n_comparisons;
            }
          }
        }
      }
    }
  }
  
  // Return performance based on measure, frobenius norm or mean absolute correlation
  if (n_comparisons == 0) {
    Rcpp::stop("cpp_bootstrap_latent_correlation: no valid block comparisons remain after feature grouping.");
  }

  return frobenius
         ? std::sqrt(acc)                 // ‖R‖_F   (same convention as solver)
         : acc / n_comparisons;           // ⟨|r|⟩
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
  std::vector<arma::mat> Xv;
  std::vector<arma::vec> Wv;
  Xv.reserve(Xin.size());
  Wv.reserve(Win.size());

  for (size_t b = 0; b < Xin.size(); ++b) {
    arma::mat Xb = Xin[b];
    arma::vec Wb = Win[b];

    if (!Xb.is_finite() || !Wb.is_finite()) {
      Rcpp::stop(std::string("Prediction-side validation received non-finite data in block ") + std::to_string(b + 1) + ".");
    }
    if (Xb.n_rows == 0 || Xb.n_cols == 0 || Wb.n_elem == 0) {
      Rcpp::stop(std::string("Prediction-side validation received an empty block or weight vector in block ") + std::to_string(b + 1) + ".");
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
    double            early_stop_threshold = 1.0,   // set <1.0 to allow early stop
    bool              permute_all_blocks   = true)  // B==2: may set false to permute only block 2
{
  const int B = X_test.size();
  if (B < 2) Rcpp::stop("Need at least 2 blocks");

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

  if (n_perm <= 0) {
    return Rcpp::List::create(
      Rcpp::_["stat_obs"] = stat_obs,
      Rcpp::_["p_value"]  = 1.0,
      Rcpp::_["n_perm"]   = 0
    );
  }

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
    if (stat_perm >= stat_obs) ++ge;

    // Early stop (optional)
    if (early_stop_threshold < 1.0 && p >= 100 && (p % 50 == 0)) {
      const double running_p = static_cast<double>(ge + 1) / static_cast<double>(p + 1);
      if (running_p > early_stop_threshold) {
        return Rcpp::List::create(
          Rcpp::_["stat_obs"] = stat_obs,
          Rcpp::_["p_value"]  = running_p,
          Rcpp::_["n_perm"]   = p + 1
        );
      }
    }
  }

  const double pval = (ge + 1.0) / (n_perm + 1.0);  // add-one smoothing
  return Rcpp::List::create(
    Rcpp::_["stat_obs"] = stat_obs,
    Rcpp::_["p_value"]  = pval,
    Rcpp::_["n_perm"]   = n_perm
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
        if (std::isfinite(r)) {
          acc += frobenius ? (r * r) : std::abs(r);
          ++pairs;
        }
      }
    }
    if (pairs == 0) {
      Rcpp::stop("cpp_bootstrap_test_oos: no valid score pairs remain for the requested statistic.");
    }
    return frobenius ? std::sqrt(acc) : (acc / pairs);
  };

  const double stat_obs = stat_from_scores(T_full, nullptr);

  if (n_boot <= 0) {
    return Rcpp::List::create(
      Rcpp::_["stat_obs"]  = stat_obs,
      Rcpp::_["boot_mean"] = NA_REAL,
      Rcpp::_["boot_se"]   = NA_REAL,
      Rcpp::_["p_value"]   = NA_REAL,
      Rcpp::_["ci_lower"]  = NA_REAL,
      Rcpp::_["ci_upper"]  = NA_REAL,
      Rcpp::_["n_boot"]    = 0
    );
  }

  arma::vec boot_vals(n_boot, arma::fill::zeros);
  int valid_reps = 0;

  for (int r = 0; r < n_boot; ++r) {
    arma::uvec idx = arma::randi<arma::uvec>(n_aligned, arma::distr_param(0, n_aligned - 1));
    double stat_boot = stat_from_scores(T_full, &idx);
    if (std::isfinite(stat_boot)) {
      boot_vals(valid_reps) = stat_boot;
      ++valid_reps;
    }
  }

  if (valid_reps == 0) {
    Rcpp::stop("cpp_bootstrap_test_oos: all bootstrap replicates produced invalid statistics.");
  }

  arma::vec vals = boot_vals.head(valid_reps);
  const double boot_mean = arma::mean(vals);
  const double boot_se = (valid_reps > 1) ? arma::stddev(vals) : 0.0;

  // p-value: fraction of bootstrap replicates with stat <= 0, testing H0: MAC <= 0.
  // Small p means the MAC is reliably positive (most bootstrap samples > 0),
  // consistent with the permutation test convention where small p = significant.
  int le_zero_count = 0;
  for (int i = 0; i < valid_reps; ++i) {
    if (vals(i) <= 0.0) ++le_zero_count;
  }
  const double p_value = (static_cast<double>(le_zero_count) + 1.0) / (static_cast<double>(valid_reps) + 1.0);

  const double conf = 1.0 - alpha;
  const double zval = R::qnorm5(1.0 - (1.0 - conf) / 2.0, 0.0, 1.0, 1, 0);
  const double ci_lower = boot_mean - zval * boot_se;
  const double ci_upper = boot_mean + zval * boot_se;

  return Rcpp::List::create(
    Rcpp::_["stat_obs"]  = stat_obs,
    Rcpp::_["boot_mean"] = boot_mean,
    Rcpp::_["boot_se"]   = boot_se,
    Rcpp::_["p_value"]   = p_value,
    Rcpp::_["ci_lower"]  = ci_lower,
    Rcpp::_["ci_upper"]  = ci_upper,
    Rcpp::_["n_boot"]    = valid_reps
  );
}
