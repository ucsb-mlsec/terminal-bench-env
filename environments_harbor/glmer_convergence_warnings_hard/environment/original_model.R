# Sleep Study Analysis - Production Script
# Multi-center clinical trial analysis
# Mixed-effects model for sleep onset latency

library(lme4)

# Load sleep study data
cat("Loading sleep study data...\n")
data <- read.csv('sleep_data.csv')

cat("Fitting mixed-effects model...\n")
cat("Sample size:", nrow(data), "observations\n")

# Fit linear mixed-effects model
# Fixed effects: demographic and clinical variables
# Random effects: random intercept for clinical site
# Model includes interaction terms to capture non-linear relationships

model <- lmer(sleep_latency ~ age + gender + bmi + baseline_score + 
              age:bmi + age:baseline_score + bmi:baseline_score +
              (1|site), 
              data = data,
              REML = TRUE)

# Extract model summary
cat("\nModel Summary:\n")
print(summary(model))

# Extract variance components
cat("\nVariance Components:\n")
vc <- VarCorr(model)
print(vc)

# Extract fixed effects
cat("\nFixed Effects:\n")
print(fixef(model))

# Model diagnostics
cat("\nModel Convergence Check:\n")
if(model@optinfo$conv$opt == 0) {
  cat("Model converged successfully\n")
} else {
  cat("WARNING: Convergence issues detected\n")
}

# Save results
site_sd <- as.numeric(attr(vc$site, "stddev"))
residual_sd <- attr(vc, "sc")
nobs <- nrow(data)

cat("\nSite SD:", site_sd, "\n")
cat("Residual SD:", residual_sd, "\n")
cat("N observations:", nobs, "\n")