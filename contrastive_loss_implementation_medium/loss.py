import numpy as np

def compute_contrastive_loss(embeddings_i, embeddings_j, temperature):
    """
    Args:
        embeddings_i: numpy array of shape (N, D) - first set of embeddings
        embeddings_j: numpy array of shape (N, D) - second set of embeddings
        temperature: float - temperature parameter for scaling
    
    Returns:
        float - the contrastive loss value
    """
    N = embeddings_i.shape[0]
    
    # Compute similarity matrices
    # Similarity between embeddings_i and embeddings_j
    sim_ij = np.matmul(embeddings_i, embeddings_j.T)
    
    # Similarity between embeddings_i and embeddings_i
    sim_ii = np.matmul(embeddings_i, embeddings_i.T)
    
    # Similarity between embeddings_j and embeddings_j
    sim_jj = np.matmul(embeddings_j, embeddings_j.T)
    
    # Scale by temperature - THIS CAUSES NUMERICAL ISSUES
    sim_ij_temp = sim_ij / temperature
    sim_ii_temp = sim_ii / temperature
    sim_jj_temp = sim_jj / temperature
    
    # Compute exponentials - OVERFLOW HAPPENS HERE
    exp_sim_ij = np.exp(sim_ij_temp)
    exp_sim_ii = np.exp(sim_ii_temp)
    exp_sim_jj = np.exp(sim_jj_temp)
    
    # Create mask to exclude self-similarities
    mask = np.eye(N, dtype=bool)
    
    loss = 0.0
    
    # Compute loss for each positive pair
    for i in range(N):
        # Positive pair: (i, i) between embeddings_i and embeddings_j
        pos_sim = exp_sim_ij[i, i]
        
        # Negative samples from embeddings_j (all except the positive)
        neg_j = np.sum(exp_sim_ij[i, :])
        
        # Negative samples from embeddings_i (excluding self)
        neg_ii = np.sum(exp_sim_ii[i, :]) - exp_sim_ii[i, i]
        
        # Negative samples from embeddings_j with itself (excluding diagonal)
        neg_jj = np.sum(exp_sim_jj[i, :]) - exp_sim_jj[i, i]
        
        # Total negatives
        denominator = neg_j + neg_ii + neg_jj
        
        # Compute log probability - UNDERFLOW/OVERFLOW ISSUES
        loss -= np.log(pos_sim / denominator)
    
    # Average over all samples
    loss = loss / N
    
    return loss