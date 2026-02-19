# Gene Expression Correlation Analysis Pipeline
# Reads gene expression CSV files and computes correlation matrices with partial correlations

# Get command line arguments
args <- commandArgs(trailingOnly=TRUE)

if (length(args) == 0) {
  stop("Usage: Rscript correlate.R <input_csv_file>")
}

input_file <- args[1]

# Print progress
cat("Processing file:", input_file, "\n")

# Read the gene expression data
expr_data <- read.csv(input_file, row.names=1, check.names=FALSE)

# Convert to numeric matrix
expr_matrix <- as.matrix(expr_data)

# Ensure all values are numeric
if (!is.numeric(expr_matrix)) {
  stop("Error: Expression data must be numeric")
}

# Transpose so genes are columns for correlation calculation
expr_t <- t(expr_matrix)

cat("Correlation matrix computed\n")

# Calculate correlation matrix between genes
cor_matrix <- cor(expr_t, use="pairwise.complete.obs")

# Invert the correlation matrix for partial correlation calculation
# WARNING: This will segfault on singular or near-singular matrices
inv_cor_matrix <- solve(cor_matrix)

# Calculate partial correlations from inverse correlation matrix
partial_cor <- -inv_cor_matrix / sqrt(outer(diag(inv_cor_matrix), diag(inv_cor_matrix)))
diag(partial_cor) <- 1

# Create output directory
output_dir <- "/workspace/output"
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

# Generate output filename
base_name <- sub("\\.csv$", "", basename(input_file))
output_file <- file.path(output_dir, paste0(base_name, "_results.csv"))

# Save partial correlation results
write.csv(partial_cor, output_file, row.names=TRUE)

cat("Results saved to:", output_file, "\n")
cat("Analysis complete\n")