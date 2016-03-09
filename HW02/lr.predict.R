lr.predict <- function(lr, data) {
  # Predicts the classes of the rows of data, based on a logistic regression
  # multinom object (from package 'nnet').  If response variable is in data,
  # remove it.
  formula <- lr$call$formula
  try({
    r <- which.response(formula, data)
    data <- data[, -r]
  }, silent = T)
  # Find probability values for each data row
  a <- 1L + (1L:length(lr$vcoefnames))
  coef <- matrix(lr$wts, nrow = lr$n[3L], byrow = TRUE)[-1L, a, drop = FALSE]
  # (above line of code is from nnet:::summary.multinom)
  X <- cbind(1, as.matrix(data))
  p <- exp(X %*% t(coef))
  baseline <- apply(p, 1, function(v) return(1/(1 + sum(v))))
  p <- baseline * p
  p <- cbind(baseline, p)
  # Find class with highest probability
  indices <- apply(p, 1, which.max)
  classes <- sapply(indices, function(i) return(lr$lev[i]))
  return(classes)
}
