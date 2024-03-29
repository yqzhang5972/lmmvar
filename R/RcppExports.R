# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' 1d score test statistics for proportion of variation
#'
#' @description {
#' compute the score test statistic for a proportion of variation of indepedent random component versus total variation in a linear mixed model,
#' assuming the covariance matrix corresponding to the random component is diagonal (see details).
#' }
#' @param h2 an numeric indicates the desired value of proportion of variation that need to be tested. Needs to be within [0,1).
#' @param y A n\eqn{\times}1 vector of observed responses.
#' @param X A n\eqn{\times}p predictors matrix, n > p.
#' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
#' @param sqRoot A boolean indicates whether to return score test statistic or +/- square root of it (\eqn{I^{-1/2}U}). Default is FALSE.
#' @return A single value showing the score test statistics at h2.
#' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
#' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
#' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
#' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
#' By the nature of the model, the support set of h2 has to be in [0,1). The test statistic follows \eqn{\Chi_1}.
#' }
#' @export
varRatioTest1d <- function(h2, y, X, lambda, sqRoot = FALSE) {
    .Call(`_lmmvar_varRatioTest1d`, h2, y, X, lambda, sqRoot)
}

#' 2d score test statistics for proportion of variation and total variation
#'
#' @description {
#' compute the score test statistic for (h2, s2p), where s2p is the total variationin a linear mixed model;
#' h2 is the proportion of variation of indepedent random component versus the total variation,
#' assuming the covariance matrix corresponding to the random component is diagonal (see details).
#' }
#' @param h2 an numeric indicates the desired value of proportion of variation that need to be tested. Needs to be within [0,1).
#' @param s2p an positive numeric indicates the desired value of total variation that need to be tested.
#' @param y A n\eqn{\times}1 vector of observed responses.
#' @param X A n\eqn{\times}p predictors matrix, n > p.
#' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
#' @return A single value showing the score test statistics
#' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
#' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
#' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
#' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
#' By the nature of the model, the support set of h2 has to be in [0,1). The test statistic follows \eqn{\Chi_2}.
#' }
#' @export
varRatioTest2d <- function(h2, s2p, y, X, lambda) {
    .Call(`_lmmvar_varRatioTest2d`, h2, s2p, y, X, lambda)
}

#' find confidence interval for h2
#'
#' @description { calculating confidence interval for h2,
#' which is the proportion of variation of indepedent random component versus the total variation in a linear mixed model.
#' Using Ternary search for finding the minimum, and binary search for finding roots.
#' }
#' @param y A n\eqn{\times}1 vector of observed responses.
#' @param X A n\eqn{\times}p predictors matrix, n > p.
#' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
#' @param range_h A vector of length 2 giving the boundary of range in which to apply searching algorithms. Needs to be within [0,1).
#' @param tolerance A positive numeric. Differences smaller than tolerance are considered to be equal.
#' @param confLevel A numeric within [0,1]. Confidence level.
#' @param maxiter A positive integer. Stop and warning if number of iterations in searching for minimum values or lower, upper bounds exceeds this value.
#' @param type A string gives whether a "two-sided", "lower_bd" (lower bound only) or "upper_bd" (upper bound only) CI needs to be calculated. Default is "two-sided".
#' @return A vector of length 2 showing confidence interval. NA if no root found.
#' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
#' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
#' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
#' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
#' By the nature of the model, the support set of h2 has to be in [0,1), and we assuming the test statistics forms a quasi-convex trend.
#' }
#' @export
confInv <- function(y, X, lambda, range_h = as.numeric( c(0.0, 1.0)), tolerance = 1e-4, confLevel = 0.95, maxiter = 50L, type = "two-sided") {
    .Call(`_lmmvar_confInv`, y, X, lambda, range_h, tolerance, confLevel, maxiter, type)
}

#' 2d score test statistics matrix for a range of proportion of variation and total variation
#'
#' @description {
#' for a range of h2 and s2p, compute the score test statistic at grid \%* grid different pair of values, where s2p is the total variationin a linear mixed model;
#' h2 is the proportion of variation of indepedent random component versus the total variation,
#' assuming the covariance matrix corresponding to the random component is diagonal (see details).
#' }
#' @param y A n\eqn{\times}1 vector of observed responses.
#' @param X A n\eqn{\times}p predictors matrix, n > p.
#' @param lambda A n\eqn{\times}1 vector represent values in the diagonal matrix Lambda. Values need to be non-negative.
#' @param range_h A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values, default is c(0, 1).
#' @param range_p A vector of length 2 giving the boundary of range of h2 which are partitioned into grid different values, default is c(0, 1).
#' @param grid An integer indicates how many different values within the range need to be tested.
#' @return A matrix showing the score test statistics.
#' @details { Assuming the linear mixed model follows \eqn{y \sim N(X\beta, \sigma_g\Lambda + \sigma_e I_n)}.
#' The proportion of variation of indepedent random component, \eqn{h^2}, is \eqn{\sigma_g / (\sigma_g+\sigma_e)},
#' the total variation \eqn{\sigma_p = \sigma_g+\sigma_e}, then y can also be seen to follow \eqn{N(X\beta, \sigma_p(h^2\Lambda + (1-h^2)I_n))}.
#' \eqn{\Lambda} is a diagonal matrix which can be achieved by applying eigen decomposition to your non-diagonal SPD \eqn{\Lambda} as well as X and y.
#' By the nature of the model, the support set of h2 has to be in [0,1), and the support set of h2 has to be positive.
#' }
#' @export
confReg <- function(y, X, lambda, range_h = as.numeric( c(0.0, 1.0)), range_p = as.numeric( c(0.0, 1.0)), grid = 200L) {
    .Call(`_lmmvar_confReg`, y, X, lambda, range_h, range_p, grid)
}

