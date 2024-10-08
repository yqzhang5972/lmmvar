#include <Rcpp.h>
// [[Rcpp::depends(Rcpp)]]

using namespace Rcpp  ;

// [[Rcpp::export]]
List RLRsimCpp (
    int p,
    int m,
    int n,
    int nsim,
    int gridlength,
    Rcpp::NumericVector mu,
    Rcpp::NumericVector lambdaGrid,
    double lambda0) {

  int is, ig, im;
  double N, D, F;
  Rcpp::NumericMatrix deno(gridlength, m) ;
  Rcpp::NumericMatrix fN(gridlength, m) ;
  Rcpp::NumericMatrix fD(gridlength, m) ;

  Rcpp::NumericVector negterm2(gridlength) ;
  Rcpp::NumericVector Chiw(m) ;
  Rcpp::IntegerVector maxIndex(nsim) ;
  Rcpp::NumericVector simDist(nsim) ;

  // without simulation
  for(ig = 0; ig < gridlength; ++ig) {
    negterm2[ig] = 0 ;
    for(im=0 ; im < m ; ++im){
      deno(ig, im) = lambdaGrid[ig] * mu[im] + 1.0 ;

      fN(ig, im) = ((lambdaGrid[ig] - lambda0) * mu[im]) / deno(ig, im) ;
      fD(ig, im) = (1 + lambda0 * mu[im]) / deno(ig, im) ;
      negterm2[ig] += log(fD(ig, im)) ;
    }
  }
  //print(negterm2) ;

  //begin simulation
  for(is = 0; is < nsim; ++is) {
    Chiw = rchisq(m, 1) ;
    simDist[is] = R_NegInf ;
    for(ig = 0; ig < gridlength; ++ig) {
      N = D = 0 ;

      for(im=0 ; im < m ; ++im){
        N += fN(ig, im) * Chiw[im] ;
        D += fD(ig, im) * Chiw[im] ;
      }
      F = (n - p) * log1p(N/D) + negterm2[ig] ;
      //Rcout<<F<<" "<<simDist[is]<<' '<<(F>= simDist[is])<<' ' ;

      if(F >= simDist[is]){
        //Rcout<<"YES"<< simDist[is]<<' ' ;
        simDist[is] = F ;
        maxIndex[is] = ig + 1 ;
      } //else break ;
    }
  }
  return List::create(Named("simDist")=simDist,
                      Named("maxIndex")=maxIndex) ;

}
