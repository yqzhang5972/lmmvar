#include <RcppEigen.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]


Eigen::MatrixXd crossProd(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  return A.transpose()*B;
}

Eigen::VectorXd rowSum(const Eigen::MatrixXd& A) {
  return A.rowwise().sum();
}



//' 1d score test statistics for proportion of variation
//'
//' @description {
//' compute the score test statistic for a proportion of variation of indepedent random component versus total variation in a linear mixed model,
//' assuming the covariance matrix corresponding to the random component is diagonal (see details).
//' }
//' @param h2 an numeric indicates the desired value of proportion of variation that need to be tested. Needs to be within [0,1).
//' @param y A n\eqn{\times}1 vector of observed responses.
//' @param X A n\eqn{\times}p predictors matrix, n > p.
//' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
//' @return A single value showing the score test statistics at h2.
//' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
//' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
//' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
//' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
//' By the nature of the model, the support set of h2 has to be in [0,1). The test statistic follows \eqn{\Chi_1}.
//' }
//' @export
// [[Rcpp::export]]
double varRatioTest1d(const double &h2, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();
  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));
  Eigen::LLT<Eigen::MatrixXd> XV2X_llt = crossProd(X, (X.array().colwise() * V2dg_inv).matrix()).llt(); // Store decomp for later inverse

  Eigen::MatrixXd betahat = XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p)) * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;
  double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n - p);

  Eigen::ArrayXd V1dg = s2phat * (lambda.array() - 1);
  Eigen::VectorXd diagH = rowSum(X.array() * (X* XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p))).array());
  Eigen::ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  Eigen::ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  double I_hh = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  double I_hp = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  double I_pp = 0.5 * pow(s2phat,-2) * (n - diagH_V2.sum());

  double score_h = 0.5 * pow(s2phat, -1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2phat,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  double test = 1 / (I_hh - pow(I_hp,2) / I_pp) * pow(score_h, 2);
  return test;
}


//' 2d score test statistics for proportion of variation and total variation
//'
//' @description {
//' compute the score test statistic for (h2, s2p), where s2p is the total variationin a linear mixed model;
//' h2 is the proportion of variation of indepedent random component versus the total variation,
//' assuming the covariance matrix corresponding to the random component is diagonal (see details).
//' }
//' @param h2 an numeric indicates the desired value of proportion of variation that need to be tested. Needs to be within [0,1).
//' @param s2p an positive numeric indicates the desired value of total variation that need to be tested.
//' @param y A n\eqn{\times}1 vector of observed responses.
//' @param X A n\eqn{\times}p predictors matrix, n > p.
//' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
//' @return A single value showing the score test statistics
//' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
//' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
//' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
//' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
//' By the nature of the model, the support set of h2 has to be in [0,1). The test statistic follows \eqn{\Chi_2}.
//' }
//' @export
// [[Rcpp::export]]
double varRatioTest2d(const double &h2, const double &s2p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();

  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));
  Eigen::LLT<Eigen::MatrixXd> XV2X_llt = crossProd(X, (X.array().colwise() * V2dg_inv).matrix()).llt();

  Eigen::ArrayXd V1dg = s2p * (lambda.array() - 1);
  Eigen::VectorXd diagH = rowSum(X.array() * (X*XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p))).array());
  Eigen::ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  Eigen::ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  Eigen::MatrixXd I(2,2);
  I(0,0) = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  I(0,1) = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  I(1,0) = I(0,1);
  I(1,1) = 0.5 * pow(s2p,-2) * (n - diagH_V2.sum());

  Eigen::MatrixXd betahat = XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p)) * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;

  Eigen::VectorXd score(2);
  score(0) = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2p,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  score(1) = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv).sum()*pow(s2p,-1) - (n-p));

  //score << score_h, score_p;
  double test = score.transpose() * I.inverse() * score;
  return test;
}


//' find confidence interval for h2
//'
//' @description { calculating confidence interval for h2,
//' which is the proportion of variation of indepedent random component versus the total variation in a linear mixed model.
//' Using Ternary search for finding the minimum, and binary search for finding roots.
//' }
//' @param range_h A vector of length 2 giving the boundary of range in which to apply searching algorithms. Needs to be within [0,1).
//' @param y A n\eqn{\times}1 vector of observed responses.
//' @param X A n\eqn{\times}p predictors matrix, n > p.
//' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
//' @param tolerance A positive numeric. Differences smaller than tolerance are considered to be equal.
//' @param confLevel A numeric within [0,1]. Confidence level.
//' @return A vector of length 2 showing confidence interval. NA if no root found.
//' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
//' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
//' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
//' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
//' By the nature of the model, the support set of h2 has to be in [0,1), and we assuming the test statistics forms a quasi-convex trend.
//' }
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector confInv(Eigen::Map<Eigen::VectorXd> range_h, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda, double tolerance = 1e-4, double confLevel = 0.95) {
  double dist = R_PosInf;
  double critPoint = R::qchisq(confLevel, 1, 1, 0);
  double lower = range_h[0];
  double upper = range_h[1];
  double a, b, test_a, test_b, test_mid;

  while (dist > tolerance) {
    a = lower + (upper-lower)/3;
    b = lower + 2* (upper-lower)/3;
    test_a = varRatioTest1d(a, y, X, lambda);
    test_b = varRatioTest1d(b, y, X, lambda);
    if (test_a < test_b) {
      upper = b;
    } else if (test_a > test_b) {
      lower = a;
    } else {
      upper = b;
      lower = a;
    }
    dist = b - a;
  }
  if (test_a > critPoint) {   // all test statistics larger than qchisq, no root
    Rcpp::NumericVector result = Rcpp::NumericVector::create(NA_REAL, NA_REAL);
    return result;
  }

  // thus lower bd of CI should be in [0,a] and upper bd should be in [a,1]
  // for lower bound
  lower = range_h[0];
  upper = b;
  double mid1 = (lower+upper) / 2;
  while (upper-lower > tolerance) {
    test_mid = varRatioTest1d(mid1, y, X, lambda) - critPoint;
    if (std::abs(test_mid) < tolerance) {
      break;
    } else if (test_mid < 0) {
      upper = mid1;
    } else {
      lower = mid1;
    }
    mid1 = (lower+upper) / 2;
  }

  // for upper bound
  lower = a;
  upper = range_h[1];
  double mid2 = (lower+upper) / 2;
  while(upper-lower > tolerance) {
    test_mid = varRatioTest1d(mid2, y, X, lambda) - critPoint;
    if (std::abs(test_mid) < tolerance) {
      break;
    } else if (test_mid < 0) {
      lower = mid2;
    } else {
      upper = mid2;
    }
    mid2 = (lower+upper) / 2;
  }
  Rcpp::NumericVector result = Rcpp::NumericVector::create(mid1, mid2);
  return result;
}

//' 2d score test statistics matrix for a range of proportion of variation and total variation
//'
//' @description {
//' for a range of h2 and s2p, compute the score test statistic at grid \%* grid different pair of values, where s2p is the total variationin a linear mixed model;
//' h2 is the proportion of variation of indepedent random component versus the total variation,
//' assuming the covariance matrix corresponding to the random component is diagonal (see details).
//' }
//' @param range_h A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values. Range needs to be within [0,1).
//' @param range_p A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values.
//' @param y A n\eqn{\times}1 vector of observed responses.
//' @param X A n\eqn{\times}p predictors matrix, n > p.
//' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
//' @param grid An integer indicates how many different values within the range need to be tested.
//' @return A matrix showing the score test statistics.
//' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
//' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
//' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
//' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
//' By the nature of the model, the support set of h2 has to be in [0,1), and the support set of h2 has to be positive.
//' }
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd confReg(Eigen::Map<Eigen::VectorXd> range_h, Eigen::Map<Eigen::VectorXd> range_p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda, int grid = 200) {
  Eigen::MatrixXd CR = Eigen::MatrixXd::Constant(grid, grid, 1);
  Eigen::VectorXd h2seq = Eigen::VectorXd::LinSpaced(grid, range_h[0], range_h[1]);
  Eigen::VectorXd s2pseq = Eigen::VectorXd::LinSpaced(grid, range_p[0], range_p[1]);

  for (int j = 0; j < grid; j++) {
    for (int k = 0; k < grid; k++) {
      CR(j, k) = varRatioTest2d(h2seq[j], s2pseq[k], y, X, lambda);
    }
  }
  return CR;
}
