set.seed(456)

# Generate synthetic gene expression data
n_genes <- 100
n_samples <- 15

# Create expression matrix with random values
expr_matrix <- matrix(rnorm(n_genes * n_samples, mean = 100, sd = 20), 
                      nrow = n_genes, 
                      ncol = n_samples)

# Add row and column names
rownames(expr_matrix) <- paste0("gene_", 1:n_genes)
colnames(expr_matrix) <- paste0("sample_", 1:n_samples)

# Introduce constant columns (zero variance genes)
# Gene 20 has constant value across all samples
expr_matrix[20, ] <- 5.0

# Gene 50 has constant value across all samples
expr_matrix[50, ] <- 10.0

# Gene 75 has constant value across all samples
expr_matrix[75, ] <- 8.5

cat("Gene expression matrix created:\n")
cat("Dimensions:", dim(expr_matrix)[1], "genes x", dim(expr_matrix)[2], "samples\n")
cat("First few values:\n")
print(expr_matrix[1:5, 1:5])

# Check for constant genes
cat("\nChecking gene variance:\n")
gene_vars <- apply(expr_matrix, 1, var)
cat("Number of genes with zero variance:", sum(gene_vars == 0), "\n")
cat("Genes with zero variance:", which(gene_vars == 0), "\n")

# Compute correlation matrix (this will create issues with constant genes)
cat("\nComputing correlation matrix...\n")
cor_matrix <- cor(t(expr_matrix))

cat("Correlation matrix dimensions:", dim(cor_matrix)[1], "x", dim(cor_matrix)[2], "\n")

# Attempt eigenvalue decomposition
cat("\nPerforming eigenvalue decomposition...\n")
eigen_result <- eigen(cor_matrix)
cat("Number of eigenvalues:", length(eigen_result$values), "\n")
cat("Top 5 eigenvalues:", eigen_result$values[1:5], "\n")

# Attempt matrix inversion using solve() - this should fail with singular matrix
cat("\nAttempting matrix inversion...\n")
inv_matrix <- solve(cor_matrix)
cat("Matrix inversion successful\n")
cat("Inverse matrix dimensions:", dim(inv_matrix)[1], "x", dim(inv_matrix)[2], "\n")

# Perform SVD on correlation matrix
cat("\nPerforming singular value decomposition...\n")
svd_result <- svd(cor_matrix)
cat("Number of singular values:", length(svd_result$d), "\n")
cat("Top 5 singular values:", svd_result$d[1:5], "\n")

# Compute covariance matrix and try inversion
cat("\nComputing covariance matrix...\n")
cov_matrix <- cov(t(expr_matrix))
cat("Covariance matrix created\n")

cat("\nAttempting covariance matrix inversion...\n")
inv_cov <- solve(cov_matrix)
cat("Covariance matrix inversion successful\n")

cat("\nAnalysis complete!\n")