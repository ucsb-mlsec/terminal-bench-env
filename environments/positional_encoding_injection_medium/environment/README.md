# Sequential Model Architecture

## Overview
This project implements a neural network for processing sequential data. The model uses an embedding layer to convert tokens into dense vector representations.

## Architecture Specifications

### Embedding Layer
- **Embedding Dimension**: 128
- **Maximum Sequence Length**: 256 tokens
- **Positional Encoding**: REQUIRED

### Positional Information
The embedding layer MUST incorporate positional information into token representations. This is achieved by adding positional encodings to the token embeddings.

Positional encodings provide the model with information about the position of each token in the sequence. Without this component, the model cannot distinguish between different orderings of the same tokens, which is critical for understanding sequential relationships.

### Implementation Requirements
- Token embeddings are generated with dimension 128
- Positional encodings must be added to token embeddings (not concatenated)
- The positional encoding should support sequences up to 256 tokens
- Both components should share the same embedding dimension (128)

This positional information is essential for the model to learn meaningful representations of sequential data.