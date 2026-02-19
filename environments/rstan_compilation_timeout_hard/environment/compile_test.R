library(rstan)

cat("Testing Stan model compilation...\n")

tryCatch({
  cat("Loading Stan model from /workspace/test_model.stan\n")
  
  model <- stan_model(file='/workspace/test_model.stan')
  
  cat("SUCCESS: Stan model compiled successfully!\n")
  cat("Model compilation completed without errors.\n")
  
  quit(status=0)
  
}, error=function(e) {
  
  cat("ERROR: Stan model compilation failed\n")
  cat("Error message:", conditionMessage(e), "\n")
  cat("Compilation process encountered errors.\n")
  
  quit(status=1)
  
})