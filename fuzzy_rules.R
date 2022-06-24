library('lfl')
main <- function(X, val, test) {
  fuzzy_X = lcut(X)
  fuzzy_val = lcut(val)
  fuzzy_test = lcut(test)
  
  rb = searchrules(fuzzy_X,
                   lhs=which(vars(fuzzy_X) != "y"),
                   rhs=which(vars(fuzzy_X) == "y"),
                   minConfidence=0.5)
  
  train_consequents = fire(fuzzy_X, rb)
  val_consequents = fire(fuzzy_val, rb)
  test_consequents = fire(fuzzy_test, rb)
  
  lhsupport = rb[[2]][,2]
  best_rule = which.max(lhsupport)
  
  train_consequents = train_consequents[[best_rule]]
  val_consequents = train_consequents[[best_rule]]
  test_consequents = train_consequents[[best_rule]]

  df <- data.frame(train_consequents, val_consequents, test_consequents)
  
  return(df)
}