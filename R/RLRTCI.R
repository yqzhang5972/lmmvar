# Take lower triangle Cholesky decomposition L of A and compute solve(A, B)
chol_solve <- function(L, B){
  forwardsolve(L, forwardsolve(L, B), transpose = TRUE)
}


#'@import Rcpp
#'@importFrom stats rchisq
#'@export RLRTSim
RLRTSim <- function(X, ZSigmasqrt, tau0 = NA, nsim = 10000, # ZSigmasqrt n*m, tcrossprod(ZSigmasqrt)=K
                    seed = NA, grid = 200) {
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

# negative log-likelihood function for new parameterization
neg_loglik2_repar <- function(par, X, y, eigens) { # par = tau, sigma^2_e
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

#'@param
#'
#'
#'
RLRTCI <- function(SimDists, ciseq, Xnew, ynew, eigens, alpha=0.05) {
  ifinCI <- rep(0, length(ciseq))
  for (ic in 1:length(ciseq)) {
    rlrt.obs <- rlrt(Xnew, ynew, eigens, ciseq[ic])
    ifinCI[ic] <- mean(rlrt.obs < SimDists[[ic]]) >= alpha
  }
  ci <- c(min(ciseq[ifinCI != 0]), max(ciseq[ifinCI != 0]))
  return(ci)
}
