# Take lower triangle Cholesky decomposition L of A and compute solve(A, B)
chol_solve <- function(L, B){
  forwardsolve(L, forwardsolve(L, B), transpose = TRUE)
}

#' Simulated distribution of the restricted likelihood ratio test statistic of variance ratio.
#'
#' @description
#' Form a simulated distribution of RLRT statistic when testing \eqn{\sigma_g^2/\sigma_e^2 = \tau_0}, see reference for method's detail.
#' @param X An n-by-p matrix of predictors, n > p.
#' @param ZSigmasqrt An n-by-m matrix with value equal to square root of covariance matrix of random component.
#' @param tau0 A scalar in \eqn{(0, \infty)} represent the value we want to test, dafault is 0.
#' @param nsim A positive integer indicating how many simulations you want.
#' @param seed A scalar to allow setting seed, default is NA.
#' @param grid A positive integer indicating how many grid you want.
#' @param tolerate A positive scalar indicating how close to boundary \eqn{\tau} needs to be.
#' @return A vector of length nsim.
#' @details
#' The function assumes the model
#' \deqn{y \sim N(X \beta, \sigma_g^2 K + \sigma_e^2 I_n),}, where \eqn{K=ZZ^T} is the covariance matrix of random component.
#' The parameter of interest is \eqn{ sigma^2_g / sigma^2_e}.
#' @import Rcpp
#' @importFrom stats rchisq
#' @references Crainiceanu, C. and Ruppert, D. (2004) Likelihood ratio tests in
#' linear mixed models with one variance component, \emph{Journal of the Royal
#' Statistical Society: Series B},\bold{66},165--185.
#' @export

RLRTSim <- function(X, ZSigmasqrt, tau0 = 0, nsim = 10000, # ZSigmasqrt n*m, tcrossprod(ZSigmasqrt)=K
                    seed = NA, grid = 200, tolerate = 1e-5) {
  n <- nrow(X)
  p <- ncol(X)
  m <- ncol(ZSigmasqrt)
  mu <- eigen(crossprod(qr.resid(qr(X), ZSigmasqrt)))$values
  #print(mu)
  if (!is.na(seed))
    set.seed(seed)

  h2seq <- seq(0, 1 - tolerate, length.out = grid)
  tauGrid <- h2seq / (1-h2seq)

  return(
    RLRsimCpp(p = p, m = m, n = n,
              nsim = nsim, mu = mu,
              tauGrid = tauGrid,
              tau0 = tau0)
    )
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
    # determinant(XSX)$modulus[1] / 2 +
    sum(log(diag(XSX_chol))) +
    sum(y^2 * Sigma_inv) / (2 * sigma2ehat)
  return(l)
}

# function to calculate observed rlrt
rlrt <- function(Xnew, ynew, eigens, tau0) {
  # Ha
  opt <- stats::optim(par=c(1,1), neg_loglik2_repar, X=Xnew, y=ynew,
                      eigens=eigens, method = "L-BFGS-B",
                      lower = c(1e-4, 1e-4),
                      upper = c(Inf, Inf)) # , control = list(trace = 10)
  # H0
  neglog <- neg_loglik1_repar(tau0, X=Xnew, y=ynew, eigens=eigens)
  rlrt <- 2 * (neglog - opt$value)
  return(rlrt[1,1])
}

#' Confidence interval for variance ratio using restricted likelihood ratio test method.
#'
#' @description
#' This function gives a confidence interval based on Simulated distribution and observed RLRT statistics, see reference for method's detail.
#' @param SimDists A vector giving the simulated distribution of RLRT statistic.
#' @param ciseq A vector values that need to be checked for \eqn{ sigma^2_g / sigma^2_e}.
#' @param ynew A vector of length n of observed responses. Value needs to be transformed so that covariance matrix of ynew is diagonal.
#' @param Xnew An n-by-p matrix of predictors, n > p. Value needs to be transformed by the same way.
#' @param eigens A vector giving eigen values of covariance matrix of random component in the model.
#' @param alpha A scalar in \eqn{(0,1)} indicating confidence level, default is 0.05.
#' @return A vector of length 2 with endpoints of the confidence interval.
#' @details
#' The function assumes the model
#' \deqn{y \sim N(X \beta, \sigma_g^2 \Lambda + \sigma_e^2 I_n),}, where \eqn{\Lambda} is a diagonal matrix with non-negative diagonal elements
#' supplied in the argument vector `lambda`. The parameter of interest is \eqn{ sigma^2_g / sigma^2_e}.
#' @references Crainiceanu, C. and Ruppert, D. (2004) Likelihood ratio tests in
#' linear mixed models with one variance component, \emph{Journal of the Royal
#' Statistical Society: Series B},\bold{66},165--185.
#' @export

RLRTCI <- function(SimDists, ciseq, Xnew, ynew, eigens, alpha = 0.05) {
  ifinCI <- rep(0, length(ciseq))
  for (ic in 1:length(ciseq)) {
    rlrt.obs <- rlrt(Xnew, ynew, eigens, ciseq[ic])
    ifinCI[ic] <- mean(rlrt.obs < SimDists[[ic]]) >= alpha
  }
  ci <- c(min(ciseq[ifinCI != 0]), max(ciseq[ifinCI != 0]))
  return(ci)
}
