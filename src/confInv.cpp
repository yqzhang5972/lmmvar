//' compute score functions for proportion of variation
//'
//' @description {
//'
//' }
//' @param range_h
//' @param y A n\%*1 vector of observed responses
//' @param X A n\%*p predictors matrix, n > p
//' @param lambda A n\%*1 vector represent values in the diagonal matrix Lambda
//' @param tolerance
//' @param confLevel description
//' @return A single value showing the score test statistics
//' @details {
//' }
//' @export


#include <RcppEigen.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]

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
