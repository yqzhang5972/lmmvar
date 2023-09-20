//' compute score functions for proportion of variation
//'
//' @description {
//'
//' }
//' @param h2 an numeric indicates the desired value of h2 that need to be tested
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


Eigen::MatrixXd crossProd(Eigen::MatrixXd A, Eigen::MatrixXd B) {
  return A.transpose()*B;
}

Eigen::VectorXd rowSum(Eigen::MatrixXd A) {
  return A.rowwise().sum();
}

// [[Rcpp::export]]
double varRatioTest1d(double h2, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda) {
  int n = X.rows();
  int p = X.cols();
  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2));

  Eigen::MatrixXd V2X = (X.array().colwise() * V2dg_inv).matrix();
  Eigen::MatrixXd XV2X = crossProd(X, V2X);
  Eigen::MatrixXd XV2X_inv = XV2X.llt().solve(Eigen::MatrixXd::Identity(p,p));

  Eigen::MatrixXd betahat = XV2X_inv * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;
  double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n-p);

  Eigen::ArrayXd V1dg = s2phat * (lambda.array()-1);
  Eigen::VectorXd diagH = rowSum(X.array() * (X*XV2X_inv).array());
  Eigen::ArrayXd diagH_V2 = diagH.array() * V2dg_inv;
  Eigen::ArrayXd V2dg_inv_V1dg = V2dg_inv * V1dg;

  double I_hh = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.pow(2).sum() - (diagH_V2 * V2dg_inv_V1dg.pow(2)).sum());
  double I_hp = 0.5 * pow(s2phat,-2) * (V2dg_inv_V1dg.sum() - (diagH_V2 * V2dg_inv_V1dg).sum());
  double I_pp = 0.5 * pow(s2phat,-2) * (n - diagH_V2.sum());
  double Iinv_hh = 1 / (I_hh - pow(I_hp,2)/I_pp);

  double score_h = 0.5 * pow(s2phat,-1) * ((ehat.array().pow(2)*V2dg_inv*V2dg_inv_V1dg).sum()*pow(s2phat,-1) + (diagH_V2*V2dg_inv_V1dg).sum() - V2dg_inv_V1dg.sum());
  double test = Iinv_hh * pow(score_h, 2);
  return test;
}
