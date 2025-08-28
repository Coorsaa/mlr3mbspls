// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// ---------- helpers -----------------------------------------------------------

// Compute one-to-many mixed-type (numeric + categorical + ordered) Gower distances.
// Numeric: range-normalized L1; Categorical: 0/1 mismatch (0=NA, -1 in test = unseen => mismatch);
// Ordered: codes scaled to [0,1], absolute difference; NA skipped pairwise.
static void gower_one_to_many_mixed(
    const arma::rowvec& x_num,        // 1 x Pn   (may have Pn = 0)
    const arma::mat&    X_num,        // N x Pn
    const arma::rowvec& ranges_num,   // 1 x Pn   (patched > 0)
    const arma::irowvec& x_cat,       // 1 x Pc   (codes: 0=NA, 1..L known, -1 unseen test level)
    const arma::imat&   X_cat,        // N x Pc
    const arma::rowvec& x_ord,        // 1 x Po   (scaled to [0,1], NaN = NA)
    const arma::mat&    X_ord,        // N x Po
    arma::vec& out_dist,              // N
    arma::uvec& out_count             // N (compared feature count)
) {
  const arma::uword N  = std::max(std::max(X_num.n_rows, X_cat.n_rows), X_ord.n_rows);
  const arma::uword Pn = X_num.n_cols;
  const arma::uword Pc = X_cat.n_cols;
  const arma::uword Po = X_ord.n_cols;

  out_dist.zeros(N);
  out_count.zeros(N);

  for (arma::uword i = 0; i < N; ++i) {
    double acc = 0.0;
    unsigned int cnt = 0;

    // numeric
    for (arma::uword j = 0; j < Pn; ++j) {
      const double a = x_num[j];
      const double b = X_num(i, j);
      if (!std::isfinite(a) || !std::isfinite(b)) continue;
      const double r = ranges_num[j];               // > 0 guaranteed
      acc += std::abs(a - b) / r;
      cnt += 1;
    }

    // categorical (unordered)
    for (arma::uword j = 0; j < Pc; ++j) {
      const int a = x_cat[j];
      const int b = X_cat(i, j);
      if (a == 0 || b == 0) continue;              // NA in either -> skip
      // 'a' can be -1 for unseen test level -> never equals any train level
      acc += (a == b) ? 0.0 : 1.0;
      cnt += 1;
    }

    // ordered (already scaled to [0,1])
    for (arma::uword j = 0; j < Po; ++j) {
      const double a = x_ord[j];
      const double b = X_ord(i, j);
      if (!std::isfinite(a) || !std::isfinite(b)) continue; // NA -> skip
      acc += std::abs(a - b);
      cnt += 1;
    }

    if (cnt > 0) {
      out_dist[i]  = acc / static_cast<double>(cnt);
      out_count[i] = cnt;
    } else {
      out_dist[i]  = arma::datum::inf;   // no overlap; invalid
      out_count[i] = 0;
    }
  }
}

// Return indices of k smallest finite values in v, requiring count >= min_cnt.
static arma::uvec k_smallest_indices(
    const arma::vec& v, const arma::uvec& count, arma::uword k, arma::uword min_cnt
) {
  arma::uvec idx_finite = arma::find_finite(v);
  arma::uvec idx_enough = arma::find(count >= min_cnt);
  arma::uvec valid      = arma::intersect(idx_finite, idx_enough);
  if (valid.n_elem == 0) return arma::uvec();

  arma::vec vsub = v.elem(valid);
  arma::uvec ord = arma::stable_sort_index(vsub, "ascend");
  arma::uword take = std::min<arma::uword>(k, ord.n_elem);
  return valid.elem(ord.head(take));
}

// ---------- classification ----------------------------------------------------

// [[Rcpp::export]]
Rcpp::List knn_predict_classif_gower_cpp(
    const arma::mat&  Xnum_train,      // N x Pn  (Pn may be 0)
    const arma::imat& Xcat_train,      // N x Pc  (codes: 0 NA; 1..L)
    const arma::mat&  Xord_train,      // N x Po  (scaled [0,1], NaN = NA)
    const arma::mat&  Xnum_test,       // M x Pn
    const arma::imat& Xcat_test,       // M x Pc  (codes: 0 NA; -1 unseen; 1..L)
    const arma::mat&  Xord_test,       // M x Po
    const arma::rowvec& ranges_num,    // 1 x Pn  (>0 or length 0)
    const Rcpp::IntegerVector& y_train, // 0..C-1
    const int k,
    const std::string& weight_scheme,  // "uniform" or "inverse"
    const double min_feature_frac,
    const Rcpp::NumericVector& priors  // length C
) {
  const arma::uword N  = std::max(std::max(Xnum_train.n_rows, Xcat_train.n_rows), Xord_train.n_rows);
  const arma::uword M  = std::max(std::max(Xnum_test.n_rows,  Xcat_test.n_rows ), Xord_test.n_rows );
  const arma::uword Pn = Xnum_train.n_cols;
  const arma::uword Pc = Xcat_train.n_cols;
  const arma::uword Po = Xord_train.n_cols;

  if (Xnum_test.n_cols != Pn || Xcat_test.n_cols != Pc || Xord_test.n_cols != Po)
    stop("Train/test feature mismatch in one of numeric/categorical/ordered blocks.");

  // number of classes C from y_train (assumed coded 0..C-1)
  int C = 0; for (int i = 0; i < y_train.size(); ++i) C = std::max(C, y_train[i] + 1);

  arma::mat prob(M, C, arma::fill::zeros);
  arma::vec d(N);
  arma::uvec cnt(N);

  const arma::uword Ptot = Pn + Pc + Po;
  arma::uword min_cnt = (Ptot > 0)
    ? static_cast<arma::uword>(std::ceil(std::max(0.0, std::min(1.0, min_feature_frac)) * static_cast<double>(Ptot)))
    : 0u;
  if (Ptot > 0 && min_cnt == 0) min_cnt = 1;

  const bool inv = (weight_scheme == "inverse");

  for (arma::uword i = 0; i < M; ++i) {
    arma::rowvec  xn = (Pn > 0) ? Xnum_test.row(i) : arma::rowvec();
    arma::irowvec xc = (Pc > 0) ? Xcat_test.row(i) : arma::irowvec();
    arma::rowvec  xo = (Po > 0) ? Xord_test.row(i) : arma::rowvec();

    gower_one_to_many_mixed(xn, Xnum_train, ranges_num, xc, Xcat_train, xo, Xord_train, d, cnt);
    arma::uvec knn_idx = k_smallest_indices(d, cnt, static_cast<arma::uword>(k), min_cnt);

    if (knn_idx.n_elem == 0) {
      // No eligible neighbor -> back off to priors
      for (int c = 0; c < C; ++c) prob(i, c) = priors[c];
      continue;
    }

    arma::vec scores(C, arma::fill::zeros);
    for (arma::uword t = 0; t < knn_idx.n_elem; ++t) {
      arma::uword j = knn_idx[t];
      const int cls = y_train[j];
      double w = 1.0;
      if (inv) {
        const double dj = d[j];
        w = (std::isfinite(dj) && dj > 1e-12) ? (1.0 / dj) : 1e12; // large weight for near-identical
      }
      scores[cls] += w;
    }

    const double ssum = arma::accu(scores);
    if (ssum <= 0 || !std::isfinite(ssum)) {
      for (int c = 0; c < C; ++c) prob(i, c) = priors[c];
    } else {
      for (int c = 0; c < C; ++c) prob(i, c) = scores[c] / ssum;
    }
  }

  return List::create(_["prob"] = prob);
}

// ---------- regression --------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List knn_predict_regr_gower_cpp(
    const arma::mat&  Xnum_train,
    const arma::imat& Xcat_train,
    const arma::mat&  Xord_train,
    const arma::mat&  Xnum_test,
    const arma::imat& Xcat_test,
    const arma::mat&  Xord_test,
    const arma::rowvec& ranges_num,
    const arma::vec&  y_train,
    const int k,
    const std::string& weight_scheme,   // "uniform" or "inverse"
    const double min_feature_frac,
    const double fallback_mean,
    const double fallback_var
) {
  const arma::uword N  = std::max(std::max(Xnum_train.n_rows, Xcat_train.n_rows), Xord_train.n_rows);
  const arma::uword M  = std::max(std::max(Xnum_test.n_rows,  Xcat_test.n_rows ), Xord_test.n_rows );
  const arma::uword Pn = Xnum_train.n_cols;
  const arma::uword Pc = Xcat_train.n_cols;
  const arma::uword Po = Xord_train.n_cols;

  if (Xnum_test.n_cols != Pn || Xcat_test.n_cols != Pc || Xord_test.n_cols != Po)
    stop("Train/test feature mismatch in one of numeric/categorical/ordered blocks.");

  arma::vec mu(M, arma::fill::zeros);
  arma::vec va(M, arma::fill::zeros);
  arma::vec d(N);
  arma::uvec cnt(N);

  const arma::uword Ptot = Pn + Pc + Po;
  arma::uword min_cnt = (Ptot > 0)
    ? static_cast<arma::uword>(std::ceil(std::max(0.0, std::min(1.0, min_feature_frac)) * static_cast<double>(Ptot)))
    : 0u;
  if (Ptot > 0 && min_cnt == 0) min_cnt = 1;

  const bool inv = (weight_scheme == "inverse");

  for (arma::uword i = 0; i < M; ++i) {
    arma::rowvec  xn = (Pn > 0) ? Xnum_test.row(i) : arma::rowvec();
    arma::irowvec xc = (Pc > 0) ? Xcat_test.row(i) : arma::irowvec();
    arma::rowvec  xo = (Po > 0) ? Xord_test.row(i) : arma::rowvec();

    gower_one_to_many_mixed(xn, Xnum_train, ranges_num, xc, Xcat_train, xo, Xord_train, d, cnt);
    arma::uvec knn_idx = k_smallest_indices(d, cnt, static_cast<arma::uword>(k), min_cnt);

    if (knn_idx.n_elem == 0) {
      mu[i] = fallback_mean;
      va[i] = fallback_var;
      continue;
    }

    arma::vec ys(knn_idx.n_elem);
    arma::vec ws(knn_idx.n_elem, arma::fill::ones);

    for (arma::uword t = 0; t < knn_idx.n_elem; ++t) {
      const arma::uword j = knn_idx[t];
      ys[t] = y_train[j];
      if (inv) {
        const double dj = d[j];
        ws[t] = (std::isfinite(dj) && dj > 1e-12) ? (1.0 / dj) : 1e12;
      }
    }

    const double wsum = arma::accu(ws);
    const double m = (wsum > 0) ? arma::dot(ws, ys) / wsum : fallback_mean;
    mu[i] = m;

    // weighted (non-unbiased) variance
    double v = 0.0;
    if (wsum > 0) {
      for (arma::uword t = 0; t < ws.n_elem; ++t) {
        const double diff = ys[t] - m;
        v += ws[t] * diff * diff;
      }
      v /= wsum;
    } else {
      v = fallback_var;
    }
    va[i] = v;
  }

  return List::create(_["mean"] = mu, _["var"] = va);
}
