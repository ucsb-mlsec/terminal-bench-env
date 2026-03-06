# Climate Research Analysis Script
# Linear regression model for temperature prediction

# Load the weather station data
weather_data <- read.csv("/workspace/weather_data.csv")

# Display basic information about the dataset
cat("Dataset dimensions:", dim(weather_data), "\n")
cat("Predictor variables: elevation, latitude, longitude, dist_coast, altitude_m, lat_degrees\n")

# Fit linear regression model using all available predictors
# Model: predict average temperature from all geographic and elevation variables
model <- lm(avg_temp ~ elevation + latitude + longitude + dist_coast + altitude_m + lat_degrees, 
            data = weather_data)

# Display model summary
summary(model)

# Extract R-squared value
r_squared <- summary(model)$r.squared

# Print results
cat("\nModel R-squared:", r_squared, "\n")
cat("Model coefficients:\n")
print(coef(model))