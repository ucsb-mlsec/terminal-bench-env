# Spatial analysis of bird survey data
# Load required library
library(mgcv)

# Read the survey data
data <- read.csv('/workspace/bird_survey.csv')

# Fit a GAM model for count data with spatial smoothing
# BUG: Using s() with two variables instead of te() for proper 2D tensor product smooth
model <- gam(count ~ s(latitude, longitude), family=poisson(), data=data)

# Extract model summary information
model_summary <- summary(model)

# Get the effective degrees of freedom for the spatial term
spatial_edf <- model_summary$s.table[1, "edf"]

# Calculate deviance explained percentage
dev_explained <- (1 - model$deviance/model$null.deviance) * 100

# Prepare output
status <- "success"
spatial_edf_rounded <- round(spatial_edf, 1)
deviance_pct_rounded <- round(dev_explained, 1)

# Write results to file
output_file <- '/workspace/model_summary.txt'
writeLines(c(
  paste0("STATUS=", status),
  paste0("SPATIAL_EDF=", spatial_edf_rounded),
  paste0("DEVIANCE_PCT=", deviance_pct_rounded)
), output_file)

# Print confirmation
cat("Model fitting complete. Results saved to", output_file, "\n")