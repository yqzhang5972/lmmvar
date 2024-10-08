library(Rcpp)
# sourceCpp("RLRTSim.cpp")

#'@import Rcpp
#'@importFrom stats rchisq
#'@export RLRTSim
RLRTSim <- function(X, ZSigmasqrt, lambda0 = NA, nsim = 10000, log.grid.hi = 8, # ZSigmasqrt n*m, tcrossprod(ZSigmasqrt)=K
                    seed = NA,
                    log.grid.lo = -10, gridlength = 200) {
  n <- nrow(X)
  p <- ncol(X)
  m <- ncol(ZSigmasqrt)
  mu <- eigen(crossprod(qr.resid(qr(X), ZSigmasqrt)))$values
  #print(mu)
  if (!is.na(seed))
    set.seed(seed)

  #generate symmetric grid around lambda0 that is log-equidistant to the right,
  make.lambdagrid <- function(lambda0, gridlength, log.grid.lo, log.grid.hi) {

    # return(c(0, exp(seq(log.grid.lo, log.grid.hi,
    #  length = gridlength - 1))))

    if (lambda0 == 0)
      return(c(0, exp(seq(log.grid.lo, log.grid.hi,
                          length = gridlength - 1))))
    else {
      leftratio <- min(max((log(lambda0)/((log.grid.hi) - (log.grid.lo))),
                           0.2), 0.8)
      leftlength <- max(round(leftratio * gridlength) - 1, 2)
      leftdistance <- lambda0 - exp(log.grid.lo)
      #make sure leftlength doesn't split the left side into too small parts:
      if (leftdistance < (leftlength * 10 * .Machine$double.eps)) {
        leftlength <- max(round(leftdistance/(10 * .Machine$double.eps)), 2)
      }
      #leftdistance approx. 1 ==> make a regular grid, since
      # (1 +- epsilon)^((1:n)/n) makes a too concentrated grid
      if (abs(leftdistance - 1) < 0.3) {
        leftgrid <- seq(exp(log.grid.lo), lambda0,
                        length = leftlength + 1)[-(leftlength + 1)]
      }
      else {
        leftdiffs <- ifelse(rep(leftdistance > 1, leftlength - 1),
                            leftdistance^((2:leftlength)/leftlength) -
                              leftdistance^(1/leftlength),
                            leftdistance^((leftlength - 1):1) -
                              leftdistance^(leftlength))
        leftgrid <- lambda0 - rev(leftdiffs)
      }
      rightlength <- gridlength - leftlength
      rightdistance <- exp(log.grid.hi) - lambda0
      rightdiffs <- rightdistance^((2:rightlength)/rightlength) -
        rightdistance^(1/rightlength)
      rightgrid <- lambda0 + rightdiffs
      return(c(0, leftgrid, lambda0, rightgrid))
    }
  }
  lambda.grid <- make.lambdagrid(lambda0, gridlength, log.grid.lo = log.grid.lo,
                                 log.grid.hi = log.grid.hi)

  res <- RLRsimCpp(p = as.integer(p), m = as.integer(m),
                   n = as.integer(n), nsim = as.integer(nsim),
                   gridlength = as.integer(gridlength),
                   mu = as.double(mu),
                   lambdaGrid = as.double(lambda.grid),
                   lambda0 = as.double(lambda0))

  return(res)
}

# negative log-likelihood function for new parameterization
neg_loglik2_repar <- function(par, X, y, eigens) { # par = lambda, sigma^2_e
  n = nrow(X)
  Sigma = (par[1]*eigens + rep(1,n)) * par[2]
  Sigma_inv = 1 / Sigma
  XSX = crossprod(X, Sigma_inv * X)
  XSX_inv = chol2inv(chol(XSX))
  betahat = XSX_inv %*% crossprod(X, Sigma_inv * y)
  l <- sum(log(Sigma)) / 2 +
    determinant(XSX)$modulus[1] / 2 +
    sum((y-X%*%betahat) * Sigma_inv * (y-X%*%betahat)) / 2
  return(l)
}

neg_loglik1_repar <- function(par, X, y, eigens) { # par = lambda0
  n = nrow(X)
  p = ncol(X)
  Sigma = (par*eigens + rep(1,n))     # \Sigma / sigma_e^2
  Sigma_inv = 1 / Sigma
  XSX = crossprod(X, Sigma_inv * X)
  XSX_inv = chol2inv(chol(XSX))
  betahat = XSX_inv %*% crossprod(X, Sigma_inv * y)
  sigma2ehat = crossprod(y-X%*%betahat, Sigma_inv * (y-X%*%betahat)) / (n-p)
  l <- sum(log(Sigma)) / 2 + (n-p) * log(sigma2ehat) / 2 +
    determinant(XSX)$modulus[1] / 2 +
    sum((y-X%*%betahat) * Sigma_inv * (y-X%*%betahat)) / 2 / sigma2ehat
  return(l)
}

# function of wald and LRT test statistics for multi-core computing, k: iteration
rlrt <- function(Xnew, ynew, eigens, lambda0) {
  # Ha
  opt = optim(par=c(1,1), neg_loglik2_repar, X=Xnew, y=ynew, eigens=eigens, method = "L-BFGS-B", lower = c(1e-4, 1e-4), upper = c(Inf, Inf)) # , control = list(trace = 10)
  # H0
  neglog <- neg_loglik1_repar(lambda0, X=Xnew, y=ynew, eigens=eigens)
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
    ifinCI[ic] <- mean(rlrt.obs < SimDists[[ic]]$simDist) >= alpha
  }
  ci <- c(min(ciseq[ifinCI != 0]), max(ciseq[ifinCI != 0]))
  return(ci)
}
