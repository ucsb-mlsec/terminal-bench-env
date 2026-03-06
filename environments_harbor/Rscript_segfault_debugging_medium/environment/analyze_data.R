# Genomic Data Analysis Script
# Reads sample data and computes summary statistics

# Read the input data
data <- read.csv('/workspace/data/samples.csv', stringsAsFactors = FALSE)

# Extract sample IDs from first column
sample_ids <- data[, 1]

# Get numeric measurement columns (all columns except the first)
numeric_cols <- data[, 2:ncol(data)]

# Initialize vectors to store results
mean_values <- numeric(nrow(data))
max_values <- numeric(nrow(data))

# Calculate statistics for each sample
for (i in 1:nrow(data)) {
  # Extract numeric values for current sample
  sample_values <- as.numeric(numeric_cols[i, ])
  
  # Calculate mean and max
  mean_values[i] <- mean(sample_values)
  max_values[i] <- max(sample_values)
}

# Create results data frame
results <- data.frame(
  sample_id = sample_ids,
  mean_value = mean_values,
  max_value = max_values
)

# Write output to CSV
write.csv(results, '/workspace/output.csv', row.names = FALSE)

print("Analysis complete!")