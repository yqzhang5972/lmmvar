#include <RcppEigen.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;               	// 'maps' rather than copies
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

MatrixXd crossProd(MatrixXd A, MatrixXd B) { //(Map<MatrixXd> A, Map<MatrixXd> B) {
  return A.transpose()*B;
}

VectorXd rowSum(MatrixXd A) {
  return A.rowwise().sum();
}


// [[Rcpp::export]]
double varRatioTest1d(double h2, Map<MatrixXd> y, Map<MatrixXd> X, Map<MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();
  ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));

  MatrixXd V2X = (X.array().colwise() * V2dg_inv).matrix();
  MatrixXd XV2X = crossProd(X, V2X);
  MatrixXd XV2X_inv = XV2X.llt().solve(MatrixXd::Identity(p,p));

  MatrixXd betahat = XV2X_inv * crossProd(X, (V2dg_inv * y.array()).matrix());
  MatrixXd ehat = y - X * betahat;
  double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n-p);

  ArrayXd V1dg = s2phat * (lambda.array()-1);
  VectorXd diagH = rowSum(X.array() * (X*XV2X_inv).array());
  ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  double I_hh = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  double I_hp = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  double I_pp = 0.5 * pow(s2phat,-2) * (n - diagH_V2.sum());
  double Iinv_hh = 1 / (I_hh - pow(I_hp,2)/I_pp);

  double score_h = 0.5 * pow(s2phat,-1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2phat,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  double test = Iinv_hh * pow(score_h, 2);
  return test;
}


// [[Rcpp::export]]
Rcpp::NumericVector confInv(Map<VectorXd> range_h, Map<MatrixXd> y, Map<MatrixXd> X, Map<MatrixXd> lambda, double tolerance = 1e-4, double confLevel = 0.95) {
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

// [[Rcpp::export]]
double varRatioTest2d(double h2, double s2p, Map<MatrixXd> y, Map<MatrixXd> X, Map<MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();

  ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));

  MatrixXd V2X = (X.array().colwise() * V2dg_inv).matrix();
  MatrixXd XV2X = crossProd(X, V2X);
  MatrixXd XV2X_inv = XV2X.llt().solve(MatrixXd::Identity(p,p));

  ArrayXd V1dg = s2p * (lambda.array()-1);
  VectorXd diagH = rowSum(X.array() * (X*XV2X_inv).array());
  ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  double I_hh = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  double I_hp = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  double I_pp = 0.5 * pow(s2p,-2) * (n - diagH_V2.sum());
  MatrixXd I(2,2);
  I << I_hh, I_hp, I_hp, I_pp;
  MatrixXd Iinv = I.inverse();

  MatrixXd betahat = XV2X_inv * crossProd(X, (V2dg_inv * y.array()).matrix());
  MatrixXd ehat = y - X * betahat;

  double score_h = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2p,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  double score_p = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv).sum()*pow(s2p,-1) - (n-p));
  VectorXd score(2);
  score << score_h, score_p;
  double test = score.transpose() * Iinv * score;
  return test;
}

// [[Rcpp::export]]
MatrixXd confReg(Map<VectorXd> range_h, Map<VectorXd> range_p, Map<MatrixXd> y, Map<MatrixXd> X, Map<MatrixXd> lambda, int grid = 200) {
  MatrixXd CR = MatrixXd::Constant(grid, grid, 1);
  VectorXd h2seq = VectorXd::LinSpaced(grid, range_h[0], range_h[1]);
  VectorXd s2pseq = VectorXd::LinSpaced(grid, range_p[0], range_p[1]);
  double test;

  for (int j = 0; j < grid; j++) {
    for (int k = 0; k < grid; k++) {
      test = varRatioTest2d(h2seq[j], s2pseq[k], X, y, lambda);
      if (test < R::qchisq(0.8, 2, 1, 0)) {
        CR(j, k) = 0.8;
      } else if (test < R::qchisq(0.9, 2, 1, 0)) {
        CR(j, k) = 0.9;
      } else if(test < R::qchisq(0.95, 2, 1, 0)) {
        CR(j, k) = 0.95;
      }
    }
  }
  return CR;
}
