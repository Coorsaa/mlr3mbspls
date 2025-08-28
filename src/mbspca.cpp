// ====================================================================
//  Group‑Sparse Multi‑Block PCA  –  C++ back‑end
//  Implements the GSMV one‑component algorithm with soft‑thresholding
// ====================================================================
//
//  Exported to R (via Rcpp):
//    * cpp_mbspca_one_lv()                  – one‑component solver
//    * perm_test_component_mbspca()         – variance‑based permutation test
//
//  Compile with:
//    Rcpp::sourceCpp("src/mbspca.cpp"), or within an R package's src/
//
#include <RcppArmadillo.h>
using arma::mat;
using arma::vec;
using arma::uvec;
using std::size_t;

// ───────────────────────── utilities (minimal) ──────────────────────
inline bool valid_vec(const vec &v)  { return v.n_elem && v.is_finite(); }
inline bool valid_mat(const mat &M)  { return M.n_rows && M.n_cols && M.is_finite(); }

inline vec l2_normalise(const vec &v)
{
  double n = arma::norm(v, 2);
  if (n < 1e-12) {
    vec z(v.n_elem, arma::fill::zeros); z(0) = 1;
    return z;
  }
  return v / n;
}

// forward declaration from mbspls.cpp (already compiled in same TU)
arma::vec soft_no_scale(const arma::vec&, double);

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

  std::vector<mat> X(B), Xt(B);
  int n = -1;
  for (int b = 0; b < B; ++b) {
    X[b]  = Rcpp::as<mat>(X_blocks[b]);
    if (!valid_mat(X[b])) Rcpp::stop("Invalid matrix in block ", b + 1);
    if (n == -1) n = X[b].n_rows;
    else if (X[b].n_rows != n)
      Rcpp::stop("All blocks must have identical row counts.");
    Xt[b] = X[b].t();
  }

  /* ── initial weights: first PCA loading per block ── */
  std::vector<vec> W(B);
  for (int b = 0; b < B; ++b) {
    arma::mat U, V; arma::vec s;
    arma::svd_econ(U, s, V, X[b]);          // economical SVD
    W[b] = l2_normalise(V.col(0));          // first principal loading
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
      W[b]  = soft_no_scale(g, c_vec(b));
    }

    /* 3) objective = variance explained */
    double num = arma::dot(t_global, t_global);
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

// ─────────────────── permutation test (variance) ────────────────────
//
// [[Rcpp::export]]
double perm_test_component_mbspca(const Rcpp::List   &X_blocks,
                                  const Rcpp::List   &W_list,
                                  const arma::vec    &c_vec,
                                  int                 n_perm = 999,
                                  double              alpha  = 0.05)
{
  const int B = X_blocks.size();
  if (!B) return NA_REAL;

  /* unpack X & W once */
  std::vector<mat> X(B);
  std::vector<vec> W(B);
  int n = -1;
  double ss_tot = 0.0;

  for (int b = 0; b < B; ++b) {
    X[b] = Rcpp::as<mat>(X_blocks[b]);
    W[b] = Rcpp::as<vec>(W_list[b]);
    if (!valid_mat(X[b])) return NA_REAL;
    if (n == -1) n = X[b].n_rows;
    ss_tot += arma::accu(arma::square(X[b]));
  }

  /* observed variance explained */
  vec t_global(n, arma::fill::zeros);
  for (int b = 0; b < B; ++b)
    t_global += X[b] * W[b];
  double var_obs = arma::dot(t_global, t_global) / ss_tot;

  /* permutation loop */
  arma::uvec idx(n);
  int ge = 0;
  for (int p = 0; p < n_perm; ++p) {
    /* permute each column of every block independently */
    std::vector<mat> Xp = X;
    for (int b = 0; b < B; ++b) {
      for (size_t j = 0; j < Xp[b].n_cols; ++j) {
        idx = arma::randperm(n);
        vec temp_col = Xp[b].col(j);  // Extract column as vector
        Xp[b].col(j) = temp_col(idx); // Permute and assign back
      }
    }
    /* refit component on permuted data */
    Rcpp::List Xp_R(B); for (int b = 0; b < B; ++b) Xp_R[b] = Xp[b];
    Rcpp::List fit = cpp_mbspca_one_lv(Xp_R, c_vec,
                                       40, 1e-4);
    vec t_perm(n, arma::fill::zeros);
    Rcpp::List Wp_R = fit["W"];
    for (int b = 0; b < B; ++b)
      t_perm += Xp[b] * Rcpp::as<vec>(Wp_R[b]);
    double var_perm = arma::dot(t_perm, t_perm) / ss_tot;
    if (var_perm >= var_obs) ++ge;

    /* early break for efficiency */
    if (p > 50 && (double)(ge + 1) / (p + 1) > alpha * 2) break;
  }

  return (ge + 1.0) / (n_perm + 1.0);
}
