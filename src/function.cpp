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
//' Compute a restricted score test-statistic for the proportion of variation due to the
//' variance component in a model with one variance component and an error term.
//' The function assumes the covariance matrix corresponding to the variance component is
//' diagonal which in practice usually means the actual covariance matrix
//' has been eigendecomposed and the transformed data are supplied to this
//' function (see details). This proportion is known as heritability in some
//' applications, and therefore denoted `h2` in the code.
//'
//' @param h2 The null hypothesis value, which needs to be in [0,1).
//' @param y A vector of length n of observed responses with diagonal covariance matrix.
//' @param X An n-by-p matrix of predictors, n > p.
//' @param lambda A vector of length n with (non-negative) variances, which are the
//' eigenvalues of the variance component covariance matrix (see details).
//' @param sqRoot If `true`, return statistic for signed square root statistic. Defaults to `false`.
//' @return The test-statistic evaluated at `h2`.
//'
//' @details
//' The function assumes the model
//' \deqn{y \sim N(X \beta, \sigma_g^2 \Lambda + \sigma_e^2 I_n),}
//' where \eqn{\Lambda} is a diagonal matrix with non-negative diagonal elements
//' supplied in the argument vector `lambda`. The parameter of interest is
//' \eqn{h^2=\sigma_g^2/(\sigma^2_g + \sigma^2_e)}.
//'
//' Usually this model results
//' from transformation: If
//' \deqn{\tilde{y} \sim N(\tilde{X} \beta, \sigma_g^2 K + \sigma_e^2 I_n),}
//' where \eqn{K} is a positive semi-definite (covariance) matrix with
//' eigendecomposition \eqn{U\Lambda U^\top}, then the transformed responses
//' \eqn{y = U^\top \tilde{y}} and predictors \eqn{X = U^\top \tilde{X}} satisfy
//' the model the function assumes.
//'
//' A linear mixed model with one random effect,
//' \eqn{\tilde{y} = \tilde{X}\beta + ZU + E}, where \eqn{U\sim N(0, \sigma^2_g I_q)}
//' and \eqn{E \sim N(0, \sigma^2_e I_n)}, is equivalent to the above with
//' \eqn{K = ZZ^\top}.
//'
//' The test-statistic is approximately chi-square with one degree of
//' freedom, even if `h2` is small or equal to zero, that is, near or at the
//' boundary of the parameter set. If `sqRoot = TRUE',
//' then the test-statistic is approximately standard normal.
//'
//' If the parameter of interest is instead \eqn{\tau = \sigma^2_g/\sigma^2_e},
//' note \eqn{h^2 = \tau / (1 + \tau)}, so the function can be evaluated the
//' null hypothesis value for \eqn{\tau}, say `tau`, by calling
//' `varRatioTest1d(h2 = tau / (1 + tau), ...)`.
//'
//' @export
// [[Rcpp::export]]
double varRatioTest1d(const double &h2, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::VectorXd> lambda, const bool sqRoot = false) {
  int n = X.rows();
  int p = X.cols();
  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2)); //O(n)
  Eigen::ArrayXd D = (lambda.array() - 1) * V2dg_inv; //O(n)

  Eigen::MatrixXd V2negsqX = (X.array().colwise() * V2dg_inv.sqrt()).matrix(); // O(np)
  Eigen::MatrixXd A_tilde = crossProd(V2negsqX, V2negsqX).llt().solve(Eigen::MatrixXd::Identity(p,p));

  Eigen::MatrixXd betahat = A_tilde * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;
  double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n - p);

  Eigen::ArrayXd diagP = rowSum(X.array() * (X * A_tilde).array()).array() * V2dg_inv;
  Eigen::ArrayXd diagQ = Eigen::ArrayXd::Constant(n,1,1) - diagP;  // diag value for (I-P) V2^(-1) V1: (1,1,...,1)T - diagP

  double I_hh = (D*D).sum() - 2 * (D*D*diagP).sum(); // tr(QDQD) = tr(D^2) - 2tr(PD^2) + tr(PDPD)
  Eigen::MatrixXd AB_tilde = A_tilde * crossProd(X, (X.array().colwise() * (V2dg_inv*D)).matrix());
  I_hh = 0.5 * (I_hh + (AB_tilde.transpose().array() * AB_tilde.array()).sum());
  //I_hh = 0.5 * (I_hh + (AB_tilde * AB_tilde).diagonal().sum());

  double score_h = 0.5 * ((ehat.array().pow(2) * V2dg_inv * D).sum()*pow(s2phat, -1) - (D * diagQ).sum());
  double I_hp = 0.5 * (D * diagQ).sum() * pow(s2phat,-1);
  double I_pp = 0.5 * (n-p) * pow(s2phat,-2);
  double test;
  if (sqRoot) {
    test = score_h / sqrt(I_hh - pow(I_hp,2) / I_pp);
  } else{
    test = pow(score_h, 2) / (I_hh - pow(I_hp,2) / I_pp);
  }
  return test;
}

//' Joint restricted score test for a proportion of variation, or heritability, and total variation
//'
//' @description
//' Compute a joint restricted score test-statistic for the total variance and the proportion
//' of variation due to the variance component in a model with one variance component and an error term.
//' See details below, and the documentation of varRatioTest1d.
//'
//' @param h2 The null hypothesis value of \eqn{h^2}, which needs to be in [0,1).
//' @param s2p The null hypothesis value of \eqn{\sigma^2_p = \sigma^2_g + \sigma^2_e}.
//' @param y A vector of length n of observed responses with diagonal covariance matrix.
//' @param X An n-by-p matrix of predictors, n > p.
//' @param lambda A vector of length n with (non-negative) variances, which are the
//' eigenvalues of the variance component covariance matrix (see details).
//' @return The test-statistic evaluated at `h2` and `s2p`.
//'
//' @details
//' The function assumes the model
//' \deqn{y \sim N(X \beta, \sigma_g^2 \Lambda + \sigma_e^2 I_n),}
//' where \eqn{\Lambda} is a diagonal matrix with non-negative diagonal elements
//' supplied in the argument vector `lambda`. See the documentation for
//' `varRatioTest1d` for how this model often results from transforming more
//' common ones using an eigendecomposition.
//'
//' The parameters in the function are \eqn{h^2=\sigma_g^2/\sigma^2_p} and
//' \eqn{\sigma^2_p = \sigma^2_g + \sigma^2_e}.
//'
//' The test-statistic is approximately chi-square with two degrees of
//' freedom, even if `h2` and `s2p` are small.
//' @export
// [[Rcpp::export]]
double varRatioTest2d(const double &h2, const double &s2p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::VectorXd> lambda) {
  int n = X.rows();
  int p = X.cols();
  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2)); //O(n)
  Eigen::ArrayXd D = (lambda.array() - 1) * V2dg_inv; //O(n)

  Eigen::MatrixXd V2negsqX = (X.array().colwise() * V2dg_inv.sqrt()).matrix(); // O(np)
  Eigen::MatrixXd A_tilde = crossProd(V2negsqX, V2negsqX).llt().solve(Eigen::MatrixXd::Identity(p,p)); // Store decomp for later , O(np^2)+llt

  Eigen::MatrixXd betahat = A_tilde * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;
  // double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n - p);

  Eigen::ArrayXd diagP = rowSum(X.array() * (X* A_tilde).array()).array() * V2dg_inv; // diag of X(X'V2X)^{-1}X' times V2^{-1}
  Eigen::ArrayXd diagQ = Eigen::ArrayXd::Constant(n,1,1) - diagP;  // diag value for (I-P)V2^(-1)^V1

  Eigen::MatrixXd I(2,2);
  Eigen::VectorXd score(2);

  I(0,0) = (D*D).sum() - 2 * (D*D*diagP).sum(); // tr(QDQD) = tr(D^2) - 2tr(PD^2) + tr(PDPD)
  Eigen::MatrixXd AB_tilde = A_tilde * crossProd(X, (X.array().colwise() * (V2dg_inv*D)).matrix());
  I(0,0) = 0.5 * (I(0,0) + (AB_tilde.transpose().array() * AB_tilde.array()).sum());


  I(0,1) = I(1,0) = 0.5 * (D * diagQ).sum() * pow(s2p,-1);
  I(1,1)= 0.5 * (n-p) * pow(s2p,-2);

  score(0) = 0.5 * ((ehat.array().pow(2) * D * V2dg_inv).sum() * pow(s2p,-1) - (D * diagQ).sum());
  score(1) = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2) * V2dg_inv).sum() * pow(s2p,-1) - (n-p));

  double test = score.transpose() * I.inverse() * score;
  return test;
}


//' Confidence interval for a proportion of variation, or heritability
//'
//' @description
//' Calculate a confidence interval by numerically inverting the test-statistic
//' implemented in the function `varRatioTest1d`. Numerical inversion is done by
//' bisection search for points where the graph of the test-statistic as a function
//' of the null-hypothesis value `h2` crosses the appropriate quantile.
//'
//' @param range_h A vector of length 2 giving the boundaries of the interval
//' within which the bisection search is performed. The endpoints must be in [0,1).
//' @param y A vector of length n of observed responses with diagonal covariance matrix.
//' @param X An n-by-p matrix of predictors, n > p.
//' @param lambda A vector of length n with (non-negative) variances, which are the
//' eigenvalues of the variance component covariance matrix (see details).
//' @param tolerance A positive scalar with the tolerance used in bisection search.
//' @param confLevel A number in (0, 1) with the level of the confidence interval.
//' @param maxiter A positive integer. Stop and warning if number of iterations
//' in search exceeds this value.
//' @param type A string that is either "two-sided", "lower_bd" (lower bound only)
//' or "upper_bd" (upper bound only). Default is "two-sided".
//'
//' @return A vector of length 2 with endpoints of the confidence interval. NA if no root found.
//' @details
//' The function assumes the model
//' \deqn{y \sim N(X \beta, \sigma_g^2 \Lambda + \sigma_e^2 I_n),}
//' where \eqn{\Lambda} is a diagonal matrix with non-negative diagonal elements
//' supplied in the argument vector `lambda`. See the documentation for
//' `varRatioTest1d` for how this model often results from transforming more
//' common ones using an eigendecomposition.
//'
//' The parameter of interest is \eqn{h^2 = \sigma^2_g / (\sigma^2_g + \sigma^2_e).}
//'
//' If the parameter of interest is instead \eqn{\tau = \sigma^2_g/\sigma^2_e},
//' note \eqn{h^2 = \tau / (1 + \tau)}. Therefore, after running the function
//' to compute interval endpoints `a < b` for \eqn{h^2}, a confidence interval for \eqn{\tau}
//' has lower endpoint `a/(1 - a)` and upper endpoint `b / (1 - b)`.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector confInv(Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::VectorXd> lambda,
                           const Rcpp::NumericVector& range_h = Rcpp::NumericVector::create(0.0, 1.0),
                           const double tolerance = 1e-4, double confLevel = 0.95, const int maxiter = 50, const std::string type = "two-sided") {
  // check if there is test-statistics within range_h
  Rcpp::NumericVector result = Rcpp::NumericVector::create(NA_REAL, NA_REAL);
  double lower = range_h[0], upper = range_h[1];
  double critPoint, test_mid, mid1, mid2;
  int iter = 0;

  if (type == "two-sided") {
    critPoint = R::qchisq(confLevel, 1, 1, 0);
    double dist = R_PosInf;
    double a, b, test_a, test_b;
    // first loop: finding a point in the CI while trying to find minimum value, assume quasi-convex.
    while (dist > tolerance) {
      a = lower + (upper-lower)/3;
      b = lower + 2* (upper-lower)/3;
      test_a = varRatioTest1d(a, y, X, lambda);
      if (test_a < critPoint) { // a is a point within the CI
        b = a;                  // find CI lower bound in (range_h[0], b) and upper bound in (a, range_h[1])
        break;
      }
      test_b = varRatioTest1d(b, y, X, lambda);
      if (test_b < critPoint) { // b is a point within the CI
        a = b;
        break;
      }
      // both test_a and test_b are greater than critPoint:
      if (test_a < test_b) {
        upper = b;
      } else if (test_a > test_b) {
        lower = a;
      } else {
        upper = b;
        lower = a;
      }
      dist = b - a;
      iter++;
      if (iter > maxiter) {
        Rcpp::warning("Warning: reached maximum iterations searching for minimum");
        break;
      }
    }
    if (test_a > critPoint) {
      if (test_b > critPoint) {  // all test-statistics larger than qchisq, no root
        Rcpp::warning("Warning: no test-statistic under threshold, returning NA");
        return result;
      }
    }
    // thus lower bd of CI should be in [range_h[0],a] and upper bd should be in [a,range_h[1]]
    // for lower bound
    lower = range_h[0];
    upper = b;
    double mid1 = (lower+upper) / 2;
    iter = 0;
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
      iter++;
      if (iter > maxiter) {
        Rcpp::warning("Warning: reached maximum iterations searching for lower bound");
        break;
      }
    }
    // for upper bound
    lower = a;
    upper = range_h[1];
    double mid2 = (lower+upper) / 2;
    iter = 0;
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
      iter++;
      if (iter > maxiter) {
        Rcpp::warning("Warning: Tolerance too low, maximum iteration of searching for upper bound tried");
        break;
      }
    }
    result = Rcpp::NumericVector::create(mid1, mid2);
    return result;
  } else if (type == "lower_bd") {
    critPoint = R::qnorm(confLevel, 0.0, 1.0, 1, 0);
    if (varRatioTest1d(range_h[1], y, X, lambda, true) > critPoint) {
      Rcpp::warning("Warning: no test-statistics in the range under threshold, return NA.");
      return result;
    }
    // upper bound is set to range_h[1], try to find lower bound
    mid1 = (lower+upper) / 2;
    iter = 0;
    while (upper-lower > tolerance) {
      test_mid = varRatioTest1d(mid1, y, X, lambda, true) - critPoint;
      if (std::abs(test_mid) < tolerance) {
        break;
      } else if (test_mid < 0) {
        upper = mid1;
      } else {
        lower = mid1;
      }
      mid1 = (lower+upper) / 2;
      iter++;
      if (iter > maxiter) {
        Rcpp::warning("Warning: Tolerance too low, maximum iteration of searching for lower bound tried");
        break;
      }
    }
    result = Rcpp::NumericVector::create(mid1, range_h[1]);
    return result;
  } else if (type == "upper_bd") {
    critPoint = R::qnorm(confLevel, 0.0, 1.0, 1, 0);
    if (varRatioTest1d(range_h[1], y, X, lambda, true) < -critPoint) {
      Rcpp::warning("Warning: no test-statistics in the range under threshold, return NA.");
      return result;
    }
    //
    mid2 = (lower+upper) / 2;
    iter = 0;
    while (upper-lower > tolerance) {
      test_mid = varRatioTest1d(mid2, y, X, lambda, true) + critPoint; // R::qnorm(1-confLevel, 0.0, 1.0, 1, 0) = -critPoint
      if (std::abs(test_mid) < tolerance) {
        break;
      } else if (test_mid < 0) {
        upper = mid2;
      } else {
        lower = mid2;
      }
      mid2 = (lower+upper) / 2;
      iter++;
      if (iter > maxiter) {
        Rcpp::warning("Warning: Tolerance too low, maximum iteration of searching for upper bound tried");
        break;
      }
    }
    result = Rcpp::NumericVector::create(range_h[0], mid2);
    return result;
  } else {
    Rcpp::warning("Warning: Please specify type as 'two-sided', 'lower_bd' or 'upper_bd'.");
    return result;
  }
}

//' Joint confidence region for proportion of variation, or heritability, and total variation
//'
//' @description
//' Calculate a confidence region by numerically inverting the test-statistic
//' implemented in the function `varRatioTest2d`. Numerical inversion is done
//' by evaluating the test-statistic on a grid (see details).
//'
//' @param range_h A vector of length 2 with the boundaries for \eqn{h^2}.
//' The endpoints must be in [0,1).
//' @param range_p A vector of length 2 giving the boundaries for \eqn{\sigma^2_p}.
//' The endpoints must be positive.
//' @param y A vector of length n of observed responses with diagonal covariance matrix.
//' @param X An n-by-p matrix of predictors, n > p.
//' @param lambda A vector of length n with (non-negative) variances, which are the
//' eigenvalues of the variance component covariance matrix (see details).
//' @param grid The number of grid points in each interval, meaning the total number
//' of points in the grid is `grid^2`.
//' @return A `grid`-by-`grid` matrix with the test-statistic evaluated at the corresponding
//' grid points. Rows index `h2`, columns index `s2p`.
//' @details
//' The function assumes the model
//' \deqn{y \sim N(X \beta, \sigma_g^2 \Lambda + \sigma_e^2 I_n),}
//' where \eqn{\Lambda} is a diagonal matrix with non-negative diagonal elements
//' supplied in the argument vector `lambda`. See the documentation for
//' `varRatioTest1d` for how this model often results from transforming more
//' common ones using an eigendecomposition.
//'
//' The parameters of interest are \eqn{h^2 = \sigma^2_g / \sigma^2_p}
//' and \eqn{\sigma^2_p = \sigma^2_g + \sigma^2_e}.
//'
//' The function creates a set of feasible values for \eqn{h^2} by taking `grid`
//' evenly spaced points in the interval defined by `range_h`. It creates
//' a set of feasible values for \eqn{\sigma^2_p} by doing the same thing
//' using the interval defined by `range_p`. The Cartesian product of these
//' two sets is the grid on which the test-statistic is evaluated. A point on
//' this grid is in the confidence region if the value of the test-statistic
//' is less than the appropriate quantile of the chi-square distribution with
//' 2 degrees of freedom.
//'
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd confReg(Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::VectorXd> lambda,
                       const Rcpp::NumericVector& range_h = Rcpp::NumericVector::create(0.0, 1.0),
                       const Rcpp::NumericVector& range_p = Rcpp::NumericVector::create(0.0, 1.0),
                       int grid = 200) {
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
