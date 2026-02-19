# Load required packages
library(caret)
library(jsonlite)

# Load the monthly sales data
# Each row represents one month of sales data (60 months total = 5 years)
sales_data <- read.csv('../data/monthly_sales.csv')

# Verify data dimensions
print(paste("Dataset dimensions:", nrow(sales_data), "months"))

# Current CV configuration - BROKEN: randomly shuffles temporal data
# Using default k-fold cross-validation which ignores temporal ordering
ctrl <- trainControl(
  method = 'cv',           # Standard cross-validation
  number = 5,              # 5 folds
  verboseIter = TRUE
)

# Problem: Future data leaks into training set
# When caret randomly shuffles rows into folds, the model can be trained on
# month 50 data to predict month 10 data. This violates the fundamental
# assumption of time-series forecasting where we can only use past data
# to predict the future.
#
# This leads to:
# - Artificially inflated performance metrics during validation
# - Model failure in production when it can't access future data
# - Incorrect feature importance estimates
# - Overly optimistic stakeholder expectations

# TODO: Fix this to use temporal ordering
# TODO: Implement time-series cross-validation with expanding window
# TODO: Load custom fold indices from solution/fold_indices.json
#
# Solution approach:
# - Use trainControl with method='cv' and custom 'index' parameter
# - Create fold indices that respect chronological order
# - Each fold should train on past data and test on future data
# - Read indices from: '../solution/fold_indices.json'

# Model training setup (currently using broken CV)
model <- train(
  sales ~ marketing_spend + promotion + competitor_count + seasonality + economic_index,
  data = sales_data,
  method = 'lm',
  trControl = ctrl
)

# Print results (will show misleadingly good performance)
print(model)
print("WARNING: These metrics are unreliable due to temporal data leakage!")