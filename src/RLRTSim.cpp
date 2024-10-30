#include <Rcpp.h>
// [[Rcpp::depends(Rcpp)]]

using namespace Rcpp  ;

// [[Rcpp::export]]
Rcpp::NumericVector RLRsimCpp (
    int p, int m, int n, int nsim,
    Rcpp::NumericVector mu, Rcpp::NumericVector tauGrid, double tau0) {

  int grid = tauGrid.length();
  Rcpp::NumericMatrix denom(grid, m), Nmat(grid, m), Dmat(grid, m);
  Rcpp::NumericVector negterm2(grid), chisq(m), simDist(nsim);
  double N, D, F;

  // without simulation
  for(int ig = 0; ig < grid; ig++) {
    negterm2[ig] = 0 ;
    for(int im=0 ; im < m ; im++){
      denom(ig, im) = tauGrid[ig] * mu[im] + 1.0;

      Nmat(ig, im) = ((tauGrid[ig] - tau0) * mu[im]) / denom(ig, im);
      Dmat(ig, im) = (1 + tau0 * mu[im]) / denom(ig, im);
      negterm2[ig] += log(Dmat(ig, im));
    }
  }
  //print(negterm2) ;

  //begin simulation
  for(int is = 0; is < nsim; is++) {
    chisq = rchisq(m, 1);
    simDist[is] = R_NegInf;

    for(int ig = 0; ig < grid; ig++) {
      N = D = 0 ;

      for(int im=0 ; im < m ; im++){
        N += Nmat(ig, im) * chisq[im];
        D += Dmat(ig, im) * chisq[im];
      }
      F = (n - p) * log1p(N/D) + negterm2[ig];
      //Rcout<<F<<" "<<simDist[is]<<' '<<(F>= simDist[is])<<' ';

      if(F >= simDist[is]){
        simDist[is] = F;
      };
    }
  }
  return simDist;

}
