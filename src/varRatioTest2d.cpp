//' compute score functions for proportion of variation
//'
//' @description {
//'
//' }
//' @param h2 an numeric indicates the desired value of h2 that need to be tested
//' @param s2p description
//' @param y A n\%*1 vector of observed responses
//' @param X A n\%*p predictors matrix, n > p
//' @param lambda A n\%*1 vector represent values in the diagonal matrix Lambda

//' @return A single value showing the score test statistics
//' @details {
//' }
//' @export


#include <RcppEigen.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]


Eigen::MatrixXd crossProd(Eigen::MatrixXd A, Eigen::MatrixXd B) { //(Map<MatrixXd> A, Map<MatrixXd> B) {
 return A.transpose()*B;
}

Eigen::VectorXd rowSum(Eigen::MatrixXd A) {
 return A.rowwise().sum();
}

// [[Rcpp::export]]
double varRatioTest2d(double h2, double s2p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();

  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));

  Eigen::MatrixXd V2X = (X.array().colwise() * V2dg_inv).matrix();
  Eigen::MatrixXd XV2X = crossProd(X, V2X);
  Eigen::MatrixXd XV2X_inv = XV2X.llt().solve(Eigen::MatrixXd::Identity(p,p));

  Eigen::ArrayXd V1dg = s2p * (lambda.array()-1);
  Eigen::VectorXd diagH = rowSum(X.array() * (X*XV2X_inv).array());
  Eigen::ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  Eigen::ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  double I_hh = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  double I_hp = 0.5 * pow(s2p,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  double I_pp = 0.5 * pow(s2p,-2) * (n - diagH_V2.sum());
  Eigen::MatrixXd I(2,2);
  I << I_hh, I_hp, I_hp, I_pp;
  Eigen::MatrixXd Iinv = I.inverse();

  Eigen::MatrixXd betahat = XV2X_inv * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;

  double score_h = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2p,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  double score_p = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv).sum()*pow(s2p,-1) - (n-p));
  Eigen::VectorXd score(2);
  score << score_h, score_p;
  double test = score.transpose() * Iinv * score;
  return test;
}
