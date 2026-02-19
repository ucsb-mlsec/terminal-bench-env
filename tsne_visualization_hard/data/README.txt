Document Embedding Dataset - README
==========================================

This dataset contains 1,500 document embeddings generated from text documents across 5 distinct topics.

Topics included in this dataset:
1. Technology - 300 documents
2. Sports - 300 documents
3. Politics - 300 documents
4. Entertainment - 300 documents
5. Science - 300 documents

Each document has been converted into a 64-dimensional embedding vector using a semantic vectorization process. These embeddings are intended to capture the semantic meaning and context of the documents such that documents discussing similar topics should be represented by vectors that are close together in the embedding space.

The embeddings are stored in vectors.npy as a numpy array of shape (1500, 64), where each row corresponds to one document. The corresponding topic labels are stored in labels.txt, with one label per line matching the order of vectors.

Expected behavior: If the vectorization process was successful, documents belonging to the same topic should form distinct, separable clusters in the embedding space. Within each cluster, documents should be more similar to each other than to documents from other topics. Clear separation between the 5 topic clusters would indicate that the embeddings have successfully captured the semantic differences between topics.