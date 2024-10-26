#include <RcppEigen.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]


Eigen::MatrixXd crossProd(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  return A.transpose()*B;
}

Eigen::VectorXd rowSum(const Eigen::MatrixXd& A) {
  return A.rowwise().sum();
}



//' Restricted score test-statistic for a proportion of variation, or heritability
//'
//' @description
//' Compute a restricted score test statistic for the proportion of variation due to the
//' variance component in a model with one variance component and an error term.
//' The function assumes the covariance matrix corresponding to the variance component is
//' diagonal which in practice usually means the actual covariance matrix
//' has been eigendecomposed and the transformed data are supplied to this
//' function (see details). This proportion is known as heritability in some
//' applications, and therefore denoted `h2` in the code.
//'
//' @param h2 The null hypothesis value, which needs to be in [0,1).
//' @param y An vector of length n of observed responses.
//' @param X An n-by-p matrix of predictors, n > p.
//' @param lambda A vector of length n with (non-negative) variances, which are the
//' eigenvalues of the variance component covariance matrix (see details).
//' @return The test-statistic evaluated at `h2`.
//'
//' @details
//' The function assumes the model
//' \deqn{y \sim N(X \beta, \sigma_g^2 K + \sigma_e^2 I_n),}
//' where \eqn{K} is a positive semi-definite (covariance) matrix with
//' eigendecomposition \eqn{U\Lambda U^\top}.
//' The parameter of interest is \eqn{h^2=\sigma_g^2/(\sigma^2_g + \sigma^2_e)}.
//' The argument `y` should be \eqn{U^\top y}, the argument `X` is \eqn{U^\top X},
//' and the argument `lambda` a vector of the diagonal elements of \eqn{\Lambda}.
//' The test statistic is approximately chi-square with one degree of
//' freedom, even if `h2` is small or equal to zero.
//'
//' If the parameter of interest is instead \eqn{\tau = \sigma^2_g/\sigma^2_e},
//' note \eqn{h^2 = \tau / (1 + \tau)}, so the function can be evaluated the
//' null hypothesis value for \eqn{\tau}, say `tau`, by calling
//' `varRatioTest1d(h2 = tau / (1 + tau), ...)`.
//'
//' @export
// [[Rcpp::export]]
double varRatioTest1d(double h2, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();
  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));

  //Eigen::MatrixXd V2X = (X.array().colwise() * V2dg_inv).matrix(); // It looks like we do not need to store this
  Eigen::MatrixXd XV2X = crossProd(X, (X.array().colwise() * V2dg_inv).matrix()); // It looks like we do not need to store this matrix
  XV2X = XV2X.llt().solve(Eigen::MatrixXd::Identity(p,p)); // Is there a way to void this inverse?

  Eigen::MatrixXd betahat = XV2X * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;
  double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n - p);

  Eigen::ArrayXd V1dg = s2phat * (lambda.array() - 1);
  Eigen::VectorXd diagH = rowSum(X.array() * (X* XV2X).array());
  Eigen::ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  Eigen::ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  double I_hh = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  double I_hp = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  double I_pp = 0.5 * pow(s2phat,-2) * (n - diagH_V2.sum());
  double Iinv_hh = 1 / (I_hh - pow(I_hp,2) / I_pp);

  double score_h = 0.5 * pow(s2phat, -1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2phat,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  double test = Iinv_hh * pow(score_h, 2);
  return test;
}


//' Joint restricted score test for a proportion of variation, or heritability, and total variation
//'
//' @description
//' Compute a joint restricted score test statistic for the total variance and the proportion
//' of variation due to the variance component in a model with one variance component and an error term.
//' See details below, and the documentation of varRatioTest1d.
//'
//' @param h2 The null hypothesis value of \eqn{h^2}, which needs to be in [0,1).
//' @param s2p The null hypothesis value of \eqn{\sigma^2_p = \sigma^2_g + \sigma^2_e}.
//' @param y An vector of length n of observed responses.
//' @param X An n-by-p matrix of predictors, n > p.
//' @param lambda A vector of length n with (non-negative) variances, which are the
//' eigenvalues of the variance component covariance matrix (see details).
//' @return The test-statistic evaluated at `h2`.
//'
//' @details
//' The function assumes the model
//' \deqn{y \sim N(X \beta, \sigma_g^2 K + \sigma_e^2 I_n),}
//' where \eqn{K} is a positive semi-definite (covariance) matrix with
//' eigendecomposition \eqn{U\Lambda U^\top}.
//' The parameters in the function are \eqn{h^2=\sigma_g^2/\sigma^2_p} and
//' \eqn{\sigma^2_p = \sigma^2_g + \sigma^2_e}.
//' The argument `y` should be \eqn{U^\top y}, the argument `X` is \eqn{U^\top X},
//' and the argument `lambda` a vector of the diagonal elements of \eqn{\Lambda}.
//' The test statistic is approximately chi-square with two degrees of
//' freedom, even if `h2` and `s2p` are small.
//' @export
// [[Rcpp::export]]
double varRatioTest2d(double h2, double s2p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();

  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));

  Eigen::MatrixXd XV2X = crossProd(X, (X.array().colwise() * V2dg_inv).matrix()); // It looks like this is never used after the next line, so let's not store it
  XV2X = XV2X.llt().solve(Eigen::MatrixXd::Identity(p,p)); // Can we avoid this inverse?

  Eigen::ArrayXd V1dg = s2p * (lambda.array() - 1);
  Eigen::VectorXd diagH = rowSum(X.array() * (X*XV2X).array());
  Eigen::ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  Eigen::ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  double I_hh = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  double I_hp = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  double I_pp = 0.5 * pow(s2p,-2) * (n - diagH_V2.sum());
  Eigen::MatrixXd I(2,2);
  I << I_hh, I_hp, I_hp, I_pp;
  Eigen::MatrixXd Iinv = I.inverse(); // Can we avoid this?

  Eigen::MatrixXd betahat = XV2X * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;

  double score_h = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2p,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  double score_p = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv).sum()*pow(s2p,-1) - (n-p));
  Eigen::VectorXd score(2);
  score << score_h, score_p;
  double test = score.transpose() * Iinv * score;
  return test;
}


//' Confidence interval for a proportion of variation, or heritability
//'
//' @description
//' Calculate a confidence interval by numerically inverting the test-statistic
//' implemented in the function `varRatioTest1d`. Numerical inversion is done by
//' bisection search of points where the graph of the test-statistic as a function
//' of the null-hypothesis value `h2` crosses the appropriate quantile.
//'
//' @param range_h A vector of length 2 giving the boundaries of the interval
//' where the bisection search is performed. The interval must be a subset of [0,1).
//' @param y An vector of length n of observed responses.
//' @param X An n-by-p matrix of predictors, n > p.
//' @param lambda A vector of length n with (non-negative) variances, which are the
//' eigenvalues of the variance component covariance matrix (see details).
//' @param tolerance A positive scalar with the tolerance used in bisection search.
//' @param confLevel A number in (0, 1) with the level of the confidence interval.
//' @return A vector of length 2 with endpoints of the confidence interval. NA if no root found.
//' @details
//' The function assumes the model
//' \deqn{y \sim N(X \beta, \sigma_g^2 K + \sigma_e^2 I_n),}
//' where \eqn{K} is a positive semi-definite (covariance) matrix with
//' eigendecomposition \eqn{U\Lambda U^\top}.
//' The parameter of interest is \eqn{h^2=\sigma_g^2/(\sigma^2_g + \sigma^2_e)}.
//' The argument `y` should be \eqn{U^\top y}, the argument `X` is \eqn{U^\top X},
//' and the argument `lambda` a vector of the diagonal elements of \eqn{\Lambda}.
//'
//' If the parameter of interest is instead \eqn{\tau = \sigma^2_g/\sigma^2_e},
//' note \eqn{h^2 = \tau / (1 + \tau)}. Therefore, after running the function
//' to compute interval endpoints `a < b` for \eqn{h^2}, a confidence interval for \eqn{\tau}
//' has lower endpoint `a/(1 - a)` and upper endpoint `b / (1 - b)`.
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
    test_a = varRatioTest1d(a, X, y, lambda);
    test_b = varRatioTest1d(b, X, y, lambda);
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
    test_mid = varRatioTest1d(mid1, X, y, lambda) - critPoint;
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
    test_mid = varRatioTest1d(mid2, X, y, lambda) - critPoint;
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
//' @description
//' for a range of h2 and s2p, compute the score test statistic at grid \%* grid different pair of values, where s2p is the total variationin a linear mixed model;
//' h2 is the proportion of variation of independent random component versus the total variation,
//' assuming the covariance matrix corresponding to the random component is diagonal (see details).
//'
//' @param range_h A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values. Range needs to be within [0,1).
//' @param range_p A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values.
//' @param y A n\%*1 vector of observed responses.
//' @param X A n\%*p predictors matrix, n > p.
//' @param lambda A n\%*1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
//' @param grid An integer indicates how many different values within the range need to be tested.
//' @return A matrix showing the score test statistics.
//' @details
//' Assuming the linear mixed model follows y ~ N(X\% beta, \% sigma_g \% Lambda + \% sigma_e I).
//' The proportion of variation of indepedent random component, h2, is \% sigma_g / (\% sigma_g+\% sigma_e),
//' the total variation \% sigma_p = \% sigma_g+\% sigma_e, then y can also be seen to follow N(X\% beta, \% sigma_p(h2\% Lambda + (1-h2)I)).
//' \% Lambda is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \% Lambda as well as X and y.
//' By the nature of the model, the support set of h2 has to be in [0,1), and the support set of h2 has to be positive.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd confReg(Eigen::Map<Eigen::VectorXd> range_h, Eigen::Map<Eigen::VectorXd> range_p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda, int grid = 200) {
  Eigen::MatrixXd CR = Eigen::MatrixXd::Constant(grid, grid, 1);
  Eigen::VectorXd h2seq = Eigen::VectorXd::LinSpaced(grid, range_h[0], range_h[1]);
  Eigen::VectorXd s2pseq = Eigen::VectorXd::LinSpaced(grid, range_p[0], range_p[1]);

  for (int j = 0; j < grid; j++) {
    for (int k = 0; k < grid; k++) {
      CR(j, k) = varRatioTest2d(h2seq[j], s2pseq[k], X, y, lambda);
    }
  }
  return CR;
}
