main <- function(X, val, test) {
  rb <- searchrule(X,
                   lhs=which(vars(d) != "y"),
                   rhs=which(vars(d) == "uptake"),
                   minConfidence=0.5)
  pbld(testing, rb, p, v, type="global")
}
