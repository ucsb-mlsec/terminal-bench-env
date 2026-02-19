set.seed(123)

# Generate synthetic gene expression data
n_genes <- 50
n_samples <- 20

# Create base gene expression matrix
gene_expr <- matrix(rnorm(n_genes * n_samples, mean = 10, sd = 2), 
                    nrow = n_genes, 
                    ncol = n_samples)

rownames(gene_expr) <- paste0("Gene_", 1:n_genes)
colnames(gene_expr) <- paste0("Sample_", 1:n_samples)

# Add perfect linear dependencies to create singular matrix issues
# Gene 5 is exact linear combination of genes 1 and 2
gene_expr[5, ] <- 2 * gene_expr[1, ] + 3 * gene_expr[2, ]

# Gene 10 is exact sum of genes 3 and 4
gene_expr[10, ] <- gene_expr[3, ] + gene_expr[4, ]

# Gene 15 is exact copy of gene 7
gene_expr[15, ] <- gene_expr[7, ]

# Gene 25 is another linear combination
gene_expr[25, ] <- 0.5 * gene_expr[8, ] - 1.5 * gene_expr[9, ]

cat("Gene expression matrix dimensions:", dim(gene_expr), "\n")

# Compute correlation matrix
cat("\nComputing correlation matrix...\n")
cor_matrix <- cor(t(gene_expr))
cat("Correlation matrix computed successfully\n")

# Perform PCA - this will fail with singular matrix
cat("\nPerforming PCA analysis...\n")
pca_result <- prcomp(t(gene_expr), center = TRUE, scale. = TRUE)

# Print PCA summary
cat("\nPCA Summary:\n")
print(summary(pca_result))

# Print proportion of variance explained by first 5 PCs
cat("\nProportion of variance explained by first 5 PCs:\n")
variance_explained <- (pca_result$sdev^2) / sum(pca_result$sdev^2)
print(variance_explained[1:5])

# Print loadings for first PC
cat("\nTop 10 gene loadings for PC1:\n")
pc1_loadings <- pca_result$rotation[, 1]
print(head(sort(abs(pc1_loadings), decreasing = TRUE), 10))

cat("\nAnalysis completed successfully!\n")