import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    """
    Basic scaled dot-product attention layer without numerical stability safeguards.
    This implementation is prone to numerical overflow for long sequences.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Tensor of shape [batch_size, seq_len, dim]
            key: Tensor of shape [batch_size, seq_len, dim]
            value: Tensor of shape [batch_size, seq_len, dim]
        
        Returns:
            output: Tensor of shape [batch_size, seq_len, dim]
        """
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        # Apply softmax to get attention weights (no numerical stability)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute weighted sum of values
        output = torch.matmul(attention_weights, value)
        
        return output