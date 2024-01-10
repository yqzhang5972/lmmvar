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
double varRatioTest1d(const double &h2, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::VectorXd> lambda) {
  int n = X.rows();
  int p = X.cols();
  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2)); //O(n)
  Eigen::MatrixXd V2negsqX = (X.array().colwise() * V2dg_inv.sqrt()).matrix(); // O(np)
  Eigen::LLT<Eigen::MatrixXd> XV2X_llt = crossProd(V2negsqX, V2negsqX).llt(); // Store decomp for later , O(np^2)+llt

  //Q = bar.A
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n,n) - V2negsqX * XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p)) * V2negsqX.transpose(); //O(pn^2)
  Eigen::ArrayXd D = (lambda.array() - 1) * V2dg_inv; //O(n)
  //Eigen::MatrixXd DQ = (Q.array().colwise() * D).matrix(); //O(n^2)

  Eigen::MatrixXd betahat = XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p)) * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;
  double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n - p);

  double traceQD = 0;
  double I_hh = 0;
  for (int i=0; i < n; i++) {
    traceQD += Q(i,i) * D(i);
    for (int j=i; j < n; j++) {
      if (i == j) {
        I_hh += Q(i,j) * Q(i,j) * D(i) * D(j) / 2;
      } else {
        I_hh += Q(i,j) * Q(i,j) * D(i) * D(j);//DQ(i,j) * DQ(j,i);
      }
    }
  }
  double I_hp = 0.5 * traceQD * pow(s2phat,-1); //0.5 * DQ.diagonal().sum() * pow(s2phat,-1);
  double I_pp = 0.5 * (n-p) * pow(s2phat,-2);
  double score_h = 0.5 * ((ehat.array().pow(2)*D*V2dg_inv).sum()*pow(s2phat,-1) - traceQD);
  double test = pow(score_h, 2) / (I_hh - pow(I_hp,2) / I_pp)  ;
  // return Rcpp::List::create(
  //   Rcpp::Named("I_hh") = I_hh, Rcpp::Named("I_hp") = I_hp, Rcpp::Named("I_pp") = I_pp,
  //               Rcpp::Named("score_h") = score_h, Rcpp::Named("test") = test,
  //               Rcpp::Named("s2phat") = s2phat, Rcpp::Named("Q") = Q, Rcpp::Named("D") = D);
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
double varRatioTest2d(const double &h2, const double &s2p, Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::VectorXd> lambda) {
  int n = X.rows();
  int p = X.cols();

  Eigen::ArrayXd V2dg_inv = 1 / (h2 * lambda.array() + (1 - h2)); //O(n)
  Eigen::MatrixXd V2negsqX = (X.array().colwise() * V2dg_inv.sqrt()).matrix(); // O(np)
  Eigen::LLT<Eigen::MatrixXd> XV2X_llt = crossProd(V2negsqX, V2negsqX).llt(); // Store decomp for later , O(np^2)+llt

  //Q = bar.A
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n,n) - V2negsqX * XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p)) * V2negsqX.transpose(); //O(n^2p)
  Eigen::ArrayXd D = (lambda.array() - 1) * V2dg_inv; //O(n)
  //Eigen::MatrixXd DQ = (Q.array().colwise() * D).matrix(); //O(n^2)

  Eigen::MatrixXd betahat = XV2X_llt.solve(Eigen::MatrixXd::Identity(p,p)) * crossProd(X, (V2dg_inv * y.array()).matrix());
  Eigen::MatrixXd ehat = y - X * betahat;
  // double s2phat = (ehat.array().pow(2) * V2dg_inv).sum() / (n - p);

  double traceQD = 0;
  Eigen::MatrixXd I(2,2);
  I(0,0) = 0;
  for (int i=0; i < n; i++) {
    traceQD += Q(i,i) * D(i);
    for (int j=i; j < n; j++) {
      if (i == j) {
        I(0,0) += Q(i,j) * Q(i,j) * D(i) * D(j) / 2;
      } else {
        I(0,0) += Q(i,j) * Q(i,j) * D(i) * D(j);//DQ(i,j) * DQ(j,i);
      }
    }
  }
  I(0,1) = 0.5 * traceQD * pow(s2p,-1); //0.5 * DQ.diagonal().sum() * pow(s2phat,-1);
  I(1,0) = I(0,1);
  I(1,1)= 0.5 * (n-p) * pow(s2p,-2);

  Eigen::VectorXd score(2);
  score(0) = 0.5 * ((ehat.array().pow(2)*D*V2dg_inv).sum()*pow(s2p,-1) - traceQD);
  score(1) = 0.5 * pow(s2p,-1) * ((ehat.array().pow(2)*V2dg_inv).sum()*pow(s2p,-1) - (n-p));

  double test = score.transpose() * I.inverse() * score;
  // return Rcpp::List::create(
  //   Rcpp::Named("I") = I,
  //   Rcpp::Named("score") = score, Rcpp::Named("test") = test,
  //   Rcpp::Named("Q") = Q, Rcpp::Named("D") = D);
  return test;
}


//' find confidence interval for h2
//'
//' @description { calculating confidence interval for h2,
//' which is the proportion of variation of indepedent random component versus the total variation in a linear mixed model.
//' Using Ternary search for finding the minimum, and binary search for finding roots.
//' }
//' @param y A n\eqn{\times}1 vector of observed responses.
//' @param X A n\eqn{\times}p predictors matrix, n > p.
//' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
//' @param range_h A vector of length 2 giving the boundary of range in which to apply searching algorithms. Needs to be within [0,1).
//' @param tolerance A positive numeric. Differences smaller than tolerance are considered to be equal.
//' @param confLevel A numeric within [0,1]. Confidence level.
//' @param maxiter A positive integer. Warning would appear if number of iterations in searching for minimum values or lower, upper bounds exceeds this value.
//' @return A vector of length 2 showing confidence interval. NA if no root found.
//' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
//' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
//' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
//' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
//' By the nature of the model, the support set of h2 has to be in [0,1), and we assuming the test statistics forms a quasi-convex trend.
//' }
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector confInv(Eigen::Map<Eigen::MatrixXd> y, Eigen::Map<Eigen::MatrixXd> X, Eigen::Map<Eigen::VectorXd> lambda,
                           const Rcpp::NumericVector& range_h = Rcpp::NumericVector::create(0.0, 1.0),
                           const double tolerance = 1e-4, const double confLevel = 0.95, const int maxiter = 50) {
 double dist = R_PosInf;
 double critPoint = R::qchisq(confLevel, 1, 1, 0);
 double lower = range_h[0];
 double upper = range_h[1];
 double a, b, test_a, test_b, test_mid;

 int iter = 0;
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
   iter++;
   if (iter > maxiter) {
     Rcpp::warning("Warning: Tolerance too low, maximum iteration of searching for minimum tried");
   }
 }
 if (test_a > critPoint & test_b > critPoint) {   // all test statistics larger than qchisq, no root
   Rcpp::NumericVector result = Rcpp::NumericVector::create(NA_REAL, NA_REAL);
   Rcpp::warning("Warning: no test statistics under threshold, return NA");
   return result;
 }

 // thus lower bd of CI should be in [0,a] and upper bd should be in [a,1]
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
     Rcpp::warning("Warning: Tolerance too low, maximum iteration of searching for lower bound tried");
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
   }
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
//' @param y A n\eqn{\times}1 vector of observed responses.
//' @param X A n\eqn{\times}p predictors matrix, n > p.
//' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
//' @param range_h A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values, default is c(0, 1).
//' @param range_p A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values, default is c(0, 1).
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
