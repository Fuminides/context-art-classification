library('lfl')
main <- function(X, X_og, val, test) {
  fuzzy_X = lcut(X)
  fuzzy_X_og = lcut(X_og)
  fuzzy_val = lcut(val)
  fuzzy_test = lcut(test)
  
  rb = searchrules(fuzzy_X,
                   lhs=which(vars(fuzzy_X) != "y"),
                   rhs=which(vars(fuzzy_X) == "y"),
                   minConfidence=0.5)
  
  train_consequents = fire(fuzzy_X, rb)
  train_consequents_og = fire(fuzzy_X_og, rb)
  val_consequents = fire(fuzzy_val, rb)
  test_consequents = fire(fuzzy_test, rb)
  
  lhsupport = rb[[2]][,2]
  best_rule = which.max(lhsupport)
  
  train_consequents = train_consequents[[best_rule]]
  train_consequents_og = train_consequents_og[[best_rule]]
  val_consequents = val_consequents[[best_rule]]
  test_consequents = test_consequents[[best_rule]]

  df <- list(train=train_consequents, trai_og=train_consequents_og, val=val_consequents, test=test_consequents)
  
  return(df)
}