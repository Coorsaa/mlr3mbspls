#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

/*
 * Ridge-capable linear regression via QR on an augmented system.
 * - X: (n x p), Y: (n x q)
 * - lambda: ridge penalty applied to all columns except:
 *    - the intercept column (assumed to be column 0 if present)
 *    - columns listed in `unpen_idx` (1-based from R)
 *
 * Pass NULL for `unpen_idx` when you have none.
 */

// [[Rcpp::export(rng=false)]]
arma::mat cpp_lm_coeff_ridge(const arma::mat& X,
                             const arma::mat& Y,
                             const double lambda,
                             Rcpp::Nullable<Rcpp::IntegerVector> unpen_idx) {
  if (X.n_rows != Y.n_rows)
    Rcpp::stop("X and Y must have the same number of rows");

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  const arma::uword q = Y.n_cols;

  if (n == 0 || p == 0 || q == 0)
    return arma::mat(p, q, arma::fill::zeros);

  // penalty per column (diag)
  arma::vec pen(p, arma::fill::value(lambda));
  // never penalize the intercept column if present at col 0
  pen(0) = 0.0;

  // ---- FIX: materialize and convert IntegerVector before subtracting 1
  if (unpen_idx.isNotNull()) {
    Rcpp::IntegerVector idx(unpen_idx.get());            // materialize sugar
    if (idx.size() > 0) {
      arma::uvec unpen = Rcpp::as<arma::uvec>(idx);      // safe conversion
      unpen -= 1;                                        // 1-based -> 0-based
      // guard indices
      unpen = unpen( unpen < p );
      pen.elem(unpen).zeros();                           // de-penalize these cols
    }
  }

  // augmented LS for ridge: [X       ] beta ≈ [Y]
  //                         [sqrt(P)]            [0]
  arma::mat Psqrt = arma::diagmat(arma::sqrt(pen));
  arma::mat Xa    = arma::join_cols(X, Psqrt);
  arma::mat Ya    = arma::join_cols(Y, arma::mat(p, q, arma::fill::zeros));

  arma::mat Q, R;
  if (!arma::qr_econ(Q, R, Xa)) {
    Rcpp::stop("cpp_lm_coeff_ridge: QR decomposition failed; cannot compute coefficients without changing the estimation method.");
  }
  const arma::mat QtY = Q.t() * Ya;
  arma::mat coef;
  const bool ok = arma::solve(coef, arma::trimatu(R), QtY, arma::solve_opts::fast);
  if (!ok || !coef.is_finite()) {
    Rcpp::stop("cpp_lm_coeff_ridge: triangular solve failed or returned non-finite coefficients.");
  }
  return coef;
}


// [[Rcpp::export(rng=false)]]
arma::mat cpp_lm_coeff(const arma::mat& X, const arma::mat& Y) {
  if (X.n_rows != Y.n_rows)
    Rcpp::stop("X and Y must have the same number of rows");
  if (X.is_empty() || Y.is_empty())
    return arma::mat(X.n_cols, Y.n_cols, arma::fill::zeros);

  arma::mat Q, R;
  if (!arma::qr_econ(Q, R, X)) {
    Rcpp::stop("cpp_lm_coeff: QR decomposition failed; cannot compute coefficients without changing the estimation method.");
  }
  const arma::mat QtY = Q.t() * Y;
  arma::mat coef;
  const bool ok = arma::solve(coef, arma::trimatu(R), QtY, arma::solve_opts::fast);
  if (!ok || !coef.is_finite()) {
    Rcpp::stop("cpp_lm_coeff: triangular solve failed or returned non-finite coefficients.");
  }
  return coef;
}
