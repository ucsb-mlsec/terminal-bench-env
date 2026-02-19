set.seed(789)

# Generate gene expression data with MORE genes than samples
# This creates a rank-deficient scenario
n_genes <- 200
n_samples <- 10

# Generate synthetic gene expression matrix (genes x samples)
gene_expression <- matrix(rnorm(n_genes * n_samples, mean = 10, sd = 2), 
                         nrow = n_genes, ncol = n_samples)
rownames(gene_expression) <- paste0("Gene_", 1:n_genes)
colnames(gene_expression) <- paste0("Sample_", 1:n_samples)

cat("Generated gene expression matrix:", n_genes, "genes x", n_samples, "samples\n")

# Calculate gene-gene correlation matrix
# This will be 200x200, but rank will be at most 10 (number of samples)
gene_cor_matrix <- cor(t(gene_expression))
cat("Calculated gene correlation matrix:", nrow(gene_cor_matrix), "x", ncol(gene_cor_matrix), "\n")

# Calculate covariance matrix between genes
gene_cov_matrix <- cov(t(gene_expression))
cat("Calculated gene covariance matrix:", nrow(gene_cov_matrix), "x", ncol(gene_cov_matrix), "\n")

# Attempt to invert the covariance matrix
# This will FAIL because the matrix is singular (rank deficient)
# With 200 genes but only 10 samples, the covariance matrix has rank <= 10
cat("Attempting to invert covariance matrix...\n")
inv_cov_matrix <- solve(gene_cov_matrix)

# Perform Mahalanobis distance calculation using the inverted covariance
# This would be used for outlier detection in gene expression space
gene_means <- rowMeans(gene_expression)
mahal_dist <- rep(0, n_samples)

for (i in 1:n_samples) {
  diff <- gene_expression[, i] - gene_means
  mahal_dist[i] <- t(diff) %*% inv_cov_matrix %*% diff
}

cat("\nMahalanobis distances for samples:\n")
print(mahal_dist)

# Identify potential outlier samples
outlier_threshold <- qchisq(0.95, df = n_genes)
outliers <- which(mahal_dist > outlier_threshold)

if (length(outliers) > 0) {
  cat("\nPotential outlier samples:", paste(colnames(gene_expression)[outliers], collapse = ", "), "\n")
} else {
  cat("\nNo outlier samples detected\n")
}

cat("\nAnalysis completed successfully!\n")