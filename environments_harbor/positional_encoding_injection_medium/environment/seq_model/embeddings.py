import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Embedding layer for sequential data processing.
    Combines token embeddings with positional encodings.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, max_position=256):
        super(TokenEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_position = max_position
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional embedding layer - creating learnable positional embeddings
        self.positional_embedding = nn.Embedding(max_position, embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        """
        Forward pass of the embedding layer.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length) containing token indices
            
        Returns:
            embeddings: Tensor of shape (batch_size, seq_length, embedding_dim)
        """
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # BUG: Positional embeddings are created but never added to token embeddings
        # The forward pass should compute positional embeddings and add them, but it doesn't!
        # 
        # What SHOULD happen:
        # seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # position_embeds = self.positional_embedding(position_ids)
        # embeddings = token_embeds + position_embeds
        
        # What ACTUALLY happens - just return token embeddings without position info:
        return token_embeds