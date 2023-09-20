//' compute score functions for proportion of variation
//'
//' @description {
//'
//' }
//' @param range_h
//' @param range_p description
//' @param y A n\%*1 vector of observed responses
//' @param X A n\%*p predictors matrix, n > p
//' @param lambda A n\%*1 vector represent values in the diagonal matrix Lambda
//' @param grid description
//' @return A single value showing the score test statistics
//' @details {
//' }
//' @export

// [[Rcpp::export]]
Eigen::MatrixXd confReg(Eigen::Map<Eigen::VectorXd> range_h, Eigen::Map<Eigen::VectorXd> range_p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::MatrixXd> lambda, int grid = 200) {
  Eigen::MatrixXd CR = Eigen::MatrixXd::Constant(grid, grid, 1);
  Eigen::VectorXd h2seq = Eigen::VectorXd::LinSpaced(grid, range_h[0], range_h[1]);
  Eigen::VectorXd s2pseq = Eigen::VectorXd::LinSpaced(grid, range_p[0], range_p[1]);
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
