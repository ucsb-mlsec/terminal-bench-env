import torch
import torch.nn as nn
import torch.nn.functional as F

class PrefixTuning(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, prefix_length=10):
        super(PrefixTuning, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.prefix_length = prefix_length
        
        # Embedding layer for vocabulary
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable prefix parameters
        self.prefix = nn.Parameter(torch.randn(prefix_length, embed_dim))
        
        # Output layer to project back to vocabulary
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids):
        # input_ids: [batch_size, seq_length]
        batch_size = input_ids.size(0)
        
        # Get embeddings for input tokens
        input_embeds = self.embedding(input_ids)  # [batch_size, seq_length, embed_dim]
        
        # Expand prefix to match batch size
        prefix_embeds = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, prefix_length, embed_dim]
        
        # Concatenate prefix with input embeddings
        combined_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)  # [batch_size, prefix_length + seq_length, embed_dim]
        
        # Apply mean pooling over the sequence dimension
        pooled = combined_embeds.mean(dim=1)  # [batch_size, embed_dim]
        
        # Project to vocabulary size
        logits = self.output(pooled)  # [batch_size, vocab_size]
        
        return logits