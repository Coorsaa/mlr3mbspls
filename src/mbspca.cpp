// ====================================================================
//  Group‑Sparse Multi‑Block PCA  –  C++ back‑end
//  Implements the GSMV one‑component algorithm with soft‑thresholding
// ====================================================================
//
//  Exported to R (via Rcpp):
//    * cpp_mbspca_one_lv()                  – one‑component solver
//    * perm_test_component_mbspca()         – cross‑block permutation test
//                                             on the variance statistic
//
//  Compile with:
//    Rcpp::sourceCpp("src/mbspca.cpp"), or within an R package's src/
//
#include <RcppArmadillo.h>
#include <limits>
using arma::mat;
using arma::vec;
using arma::uvec;
using std::size_t;

// ───────────────────────── utilities (minimal) ──────────────────────
inline bool valid_vec(const vec &v)  { return v.n_elem && v.is_finite(); }
inline bool valid_mat(const mat &M)  { return M.n_rows && M.n_cols && M.is_finite(); }

// Shared constrained PMD update from the linked mbspls.cpp translation unit.
arma::vec pmd_update_bisection(const arma::vec&, double, int, double);

// ───────────────────────── one‑LV solver ────────────────────────────
//
// [[Rcpp::export]]
Rcpp::List cpp_mbspca_one_lv(const Rcpp::List   &X_blocks,
                             const arma::vec    &c_vec,
                             int                 max_iter = 50,
                             double              tol      = 1e-4)
{
  const int B = X_blocks.size();
  if (!B) Rcpp::stop("X_blocks is empty.");
  if (static_cast<int>(c_vec.n_elem) != B)
    Rcpp::stop("c_vec length must equal number of blocks");
  if (!c_vec.is_finite() || arma::any(c_vec <= 0.0))
    Rcpp::stop("c_vec must contain finite, strictly positive constraints.");
  if (max_iter < 1)
    Rcpp::stop("max_iter must be at least 1.");
  if (!std::isfinite(tol) || tol <= 0.0)
    Rcpp::stop("tol must be finite and positive.");

  std::vector<mat> X(B), Xt(B);
  int n = -1;
  for (int b = 0; b < B; ++b) {
    X[b]  = Rcpp::as<mat>(X_blocks[b]);
    if (!valid_mat(X[b]))
      Rcpp::stop("Invalid matrix in block %d.", b + 1);
    if (n == -1) n = X[b].n_rows;
    else if (X[b].n_rows != n)
      Rcpp::stop("All blocks must have identical row counts.");
    const double upper = std::sqrt(static_cast<double>(X[b].n_cols));
    if (c_vec(b) < 1.0 || c_vec(b) > upper + 1e-10)
      Rcpp::stop("c_vec entry for block %d must lie in [1, sqrt(p)].", b + 1);
    Xt[b] = X[b].t();
  }
  if (n < 3) Rcpp::stop("At least 3 rows are required.");

  /* ── initial weights: first PCA loading per block ── */
  std::vector<vec> W(B);
  vec initial_score(n, arma::fill::zeros);
  for (int b = 0; b < B; ++b) {
    arma::mat U, V; arma::vec s;
    if (!arma::svd_econ(U, s, V, X[b]) || !s.n_elem || s(0) < 1e-12) {
      Rcpp::stop("Cannot initialize MB-sPCA from degenerate block %d.", b + 1);
    }
    W[b] = pmd_update_bisection(V.col(0), c_vec(b), 80, 1e-8);
    vec score = X[b] * W[b];
    // SVD signs are arbitrary. Orient initial scores before summing them,
    // otherwise perfectly associated blocks can cancel to a zero gradient.
    if (arma::dot(initial_score, score) < 0.0) {
      W[b] *= -1.0;
      score *= -1.0;
    }
    initial_score += score;
  }

  double obj_old = -1.0;
  bool converged = false;

  for (int it = 0; it < max_iter; ++it) {

    /* 1) block scores + global score */
    std::vector<vec> t_block(B);
    vec t_global(n, arma::fill::zeros);
    for (int b = 0; b < B; ++b) {
      t_block[b] = X[b] * W[b];
      t_global  += t_block[b];
    }

    /* 2) update weights */
    for (int b = 0; b < B; ++b) {
      vec g = Xt[b] * t_global;               // gradient
      W[b] = pmd_update_bisection(g, c_vec(b), 80, 1e-8);
    }

    /* 3) objective at the updated weights = variance explained */
    vec t_updated(n, arma::fill::zeros);
    for (int b = 0; b < B; ++b)
      t_updated += X[b] * W[b];
    double num = arma::dot(t_updated, t_updated);
    double denom = 0.0;
    for (int b = 0; b < B; ++b)
      denom += arma::accu(arma::square(X[b]));
    double obj = (denom < 1e-12) ? 0.0 : num / denom;

    if (std::abs(obj - obj_old) < tol) { converged = true; break; }
    obj_old = obj;
  }

  /* expose result */
  Rcpp::List W_out(B);
  for (int b = 0; b < B; ++b) W_out[b] = W[b];

  return Rcpp::List::create(
    Rcpp::_["W"]         = W_out,
    Rcpp::_["converged"] = converged
  );
}

// ──────────── cross-block permutation test (variance statistic) ─────────
//
// Statistic: variance explained by the refitted component,
// ||sum_b X_b w_b||^2 / SS_tot. Null: the blocks are mutually independent.
// Rows are permuted independently within each block, which keeps every
// within-block covariance (and so each block's own ||X_b w_b||^2 maximum)
// and destroys only the cross-block alignment. A component that carries
// block-specific variance but no cross-block association is therefore not
// significant. A single block is invariant under row permutation, so at
// least two blocks are required.
//
// [[Rcpp::export]]
double perm_test_component_mbspca(const Rcpp::List   &X_blocks,
                                  const Rcpp::List   &W_list,
                                  const arma::vec    &c_vec,
                                  int                 n_perm    = 999,
                                  double              alpha     = 0.05,
                                  int                 max_iter  = 50,
                                  double              tol       = 1e-4)
{
  const int B = X_blocks.size();
  if (!B) Rcpp::stop("perm_test_component_mbspca: X_blocks is empty.");
  if (B < 2)
    Rcpp::stop("perm_test_component_mbspca: the cross-block row-permutation null requires at least two blocks; a single block is invariant under row permutation.");
  if (W_list.size() != B)
    Rcpp::stop("perm_test_component_mbspca: W_list length must equal the number of blocks.");
  if (static_cast<int>(c_vec.n_elem) != B || !c_vec.is_finite() || arma::any(c_vec <= 0.0))
    Rcpp::stop("perm_test_component_mbspca: c_vec must contain one finite, strictly positive value per block.");
  if (n_perm < 1)
    Rcpp::stop("perm_test_component_mbspca: n_perm must be at least 1.");
  if (max_iter < 1)
    Rcpp::stop("perm_test_component_mbspca: max_iter must be at least 1.");
  if (!std::isfinite(tol) || tol <= 0.0)
    Rcpp::stop("perm_test_component_mbspca: tol must be finite and positive.");
  (void)alpha;  // Retained for API compatibility; every requested permutation is used.

  /* unpack X & W once */
  std::vector<mat> X(B);
  std::vector<vec> W(B);
  int n = -1;
  double ss_tot = 0.0;

  for (int b = 0; b < B; ++b) {
    X[b] = Rcpp::as<mat>(X_blocks[b]);
    W[b] = Rcpp::as<vec>(W_list[b]);
    if (!valid_mat(X[b])) Rcpp::stop(std::string("perm_test_component_mbspca: invalid matrix in block ") + std::to_string(b + 1) + ".");
    if (!valid_vec(W[b]) || W[b].n_elem != X[b].n_cols)
      Rcpp::stop(std::string("perm_test_component_mbspca: invalid or dimensionally incompatible weight vector in block ") + std::to_string(b + 1) + ".");
    if (n == -1) n = X[b].n_rows;
    else if (X[b].n_rows != static_cast<arma::uword>(n))
      Rcpp::stop("perm_test_component_mbspca: all blocks must have identical row counts.");
    ss_tot += arma::accu(arma::square(X[b]));
  }
  if (n < 3)
    Rcpp::stop("perm_test_component_mbspca: at least 3 rows are required.");
  if (!std::isfinite(ss_tot) || ss_tot <= 1e-12)
    Rcpp::stop("perm_test_component_mbspca: total sum of squares is numerically zero.");

  /* observed variance explained */
  vec t_global(n, arma::fill::zeros);
  for (int b = 0; b < B; ++b)
    t_global += X[b] * W[b];
  double var_obs = arma::dot(t_global, t_global) / ss_tot;

  /* permutation loop */
  int ge = 0;
  for (int p = 0; p < n_perm; ++p) {
    /* permute rows of each block independently: within-block covariance is
     * preserved and only the cross-block alignment is destroyed */
    std::vector<mat> Xp(B);
    for (int b = 0; b < B; ++b) {
      arma::uvec row_idx = arma::randperm(n);
      Xp[b] = X[b].rows(row_idx);
    }
    /* refit component on permuted data */
    Rcpp::List Xp_R(B); for (int b = 0; b < B; ++b) Xp_R[b] = Xp[b];
    Rcpp::List fit = cpp_mbspca_one_lv(Xp_R, c_vec,
                                       max_iter, tol);
    vec t_perm(n, arma::fill::zeros);
    Rcpp::List Wp_R = fit["W"];
    for (int b = 0; b < B; ++b)
      t_perm += Xp[b] * Rcpp::as<vec>(Wp_R[b]);
    double var_perm = arma::dot(t_perm, t_perm) / ss_tot;
    if (var_perm >= var_obs - 100.0 * std::numeric_limits<double>::epsilon() * std::abs(var_obs)) ++ge;

  }

  return (ge + 1.0) / (n_perm + 1.0);
}
