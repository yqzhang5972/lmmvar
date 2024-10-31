# Take lower triangle Cholesky decomposition L of A and compute solve(A, B)
chol_solve <- function(L, B){
  forwardsolve(L, forwardsolve(L, B), transpose = TRUE)
}

#' Simulate the distribution of the restricted likelihood ratio test statistic for a variance ratio.
#'
#' @description
#' Simulate the distribution of RLRT statistic in a variance component model under the null hypothesis that that \eqn{\tau = \sigma_g^2/\sigma_e^2 = \mathtt{tau0}} (see details and references).
#' 
#' @param X An n-by-p matrix of predictors, n > p.
#' @param Ksqrt An n-by-m matrix that is a square-root of the variance component covariance matrix (see details).
#' @param tau0 A scalar null hyothesis in \eqn{[0, \infty)}, default is 0.
#' @param nsim A positive integer indicating the number of simulations.
#' @param seed A scalar to allow setting seed, default is NA.
#' @param grid A positive integer with the number of grid points at which to evaluate when inverting the test-statistic to get a confidence region.
#' @param tol A positive scalar such that \eqn{h^2 = \tau / (1 + \tau) < 1 - \mathtt{tol}}.
#' @return A vector of length `nsim` of simulated values of the test-statistic.
#' @details
#' The function assumes the model
#' \deqn{y \sim N(X \beta, \sigma_g^2 K + \sigma_e^2 I_n),} where \eqn{K} is
#' the variance component covariance matrix. The matrix `Ksqrt` should be such
#' that `tcrossprod(Ksqrt)` is equal to \eqn{K}.
#' 
#' The parameter of interest is \eqn{\tau = \sigma^2_g / \sigma^2_e}.
#' 
#' An implementation of this test-statistic for the case where `tau0 = 0`
#' is also available in the package `RLRsim`.
#' 
#' @import Rcpp
#' @importFrom stats rchisq
#' @references Crainiceanu, C. and Ruppert, D. (2004) Likelihood ratio tests in
#' linear mixed models with one variance component, \emph{Journal of the Royal
#' Statistical Society: Series B},\bold{66},165--185.
#' @export

simulate_RLRT <- function(X, Ksqrt, tau0 = 0, nsim = 10000,
                    seed = NA, grid = 200, tol = 1e-5) {
  n <- nrow(X)
  p <- ncol(X)
  m <- ncol(Ksqrt)
  mu <- eigen(crossprod(qr.resid(qr(X), Ksqrt)))$values
  if (!is.na(seed)) set.seed(seed)

  h2seq <- seq(0, 1 - tol, length.out = grid)
  tauGrid <- h2seq / (1 - h2seq)

  RLRsimCpp(p = p, m = m, n = n,
              nsim = nsim, mu = mu,
              tauGrid = tauGrid,
              tau0 = tau0)
}

# negative log-likelihood function for new parameterization (tau, sigma^2_e)^T
neg_loglik2_repar <- function(par, X, y, eigens) {
  n <- nrow(X)
  Sigma <- (par[1] * eigens + rep(1, n)) * par[2]
  Sigma_inv <- 1 / Sigma
  XSX <- crossprod(X, Sigma_inv * X)
  XSX_chol <- chol(XSX)
  betahat <- chol_solve(t(XSX_chol), crossprod(X, Sigma_inv * y))
  y <- y - X %*% betahat
  l <- sum(log(Sigma)) / 2 + sum(log(diag(XSX_chol))) + sum(y^2 * Sigma_inv) / 2
  return(l)
}

# negative log-likelihood function for tau
neg_loglik1_repar <- function(par, X, y, eigens) { # par = tau0
  n <- nrow(X)
  p <- ncol(X)
  Sigma <- (par * eigens + rep(1, n))     # \Sigma / sigma_e^2
  Sigma_inv <- 1 / Sigma
  XSX <- crossprod(X, Sigma_inv * X)
  XSX_chol <- chol(XSX)
  betahat <- chol_solve(t(XSX_chol), crossprod(X, Sigma_inv * y))
  y <- y - X %*% betahat
  sigma2ehat <- crossprod(y, Sigma_inv * y) / (n - p)
  l <- sum(log(Sigma)) / 2 + (n - p) * log(sigma2ehat) / 2 +
    sum(log(diag(XSX_chol))) +
    sum(y^2 * Sigma_inv) / (2 * sigma2ehat)
  return(l)
}

# function to calculate observed rlrt
rlrt <- function(Xnew, ynew, eigens, tau0, tol = 1e-4) {
  # Ha
  opt <- stats::optim(par=c(1,1), neg_loglik2_repar, X=Xnew, y=ynew,
                      eigens=eigens, method = "L-BFGS-B",
                      lower = c(tol, tol),
                      upper = c(Inf, Inf))
  # H0
  neglog <- neg_loglik1_repar(tau0, X=Xnew, y=ynew, eigens=eigens)
  rlrt <- 2 * (neglog - opt$value)
  return(rlrt[1,1])
}

#' Confidence interval for a variance ratio by inverting a restricted likelihood ratio test.
#'
#' @description
#' This function computes a confidence interval for \eqn{\tau = \sigma^2_g / \sigma^2_e} (see details) by inverting the restricted
#' likelihood ratio test-statistic using the sample quantiles from parametric bootstrap samples of the test-statistics under the null hypothesis.
#' @param SimDists A list with simulated RLRT statistics for
#' each velement in `ciseq`. Usually each element in `SimDists` is a vector returned by running the function
#' `simulate_RLRT`.
#' @param ciseq A vector of potential values of \eqn{\tau} to be included in the confidence interval; the jth element in this vector corresponds to the jth
#' element of `SimDists`.
#' @param y A vector of length n of observed responses.
#' @param X An n-by-p matrix of predictors, n > p. 
#' @param lambda A vector with eigenvalues of the variance component covariance
#' matrix (see details).
#' @param alpha A scalar in \eqn{(0,1)} indicating confidence level, default is 0.05.
#' @return A vector of length 2 with endpoints of the confidence interval.
#' @details
#' The function assumes the model
#' \deqn{y \sim N(X \beta, \sigma_g^2 \Lambda + \sigma_e^2 I_n),} where \eqn{\Lambda} is a diagonal matrix with non-negative diagonal elements
#' supplied in the argument vector `lambda`.
#' 
#' The parameter of interest is \eqn{\tau =  \sigma^2_g / \sigma^2_e}.
#' 
#' See documention of `varTestRatio1d` for how this model often results from transforming more common ones using an eigendecomposition.
#' 
#' @references Crainiceanu, C. and Ruppert, D. (2004) Likelihood ratio tests in
#' linear mixed models with one variance component, \emph{Journal of the Royal
#' Statistical Society: Series B},\bold{66},165--185.
#' @export

RLRTCI <- function(SimDists, ciseq, X, y, lambda, alpha = 0.05) {
  ifinCI <- rep(0, length(ciseq))
  for (ic in 1:length(ciseq)) {
    rlrt.obs <- rlrt(X, y, lambda, ciseq[ic])
    ifinCI[ic] <- mean(rlrt.obs < SimDists[[ic]]) >= alpha
  }
  ci <- c(min(ciseq[ifinCI != 0]), max(ciseq[ifinCI != 0]))
  return(ci)
}
