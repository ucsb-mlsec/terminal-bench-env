#!/usr/bin/env python3

import os
import json
import random

# Configuration
VOCAB_PATH = "/workspace/vocab.txt"
CORPUS_PATH = "/workspace/data/corpus.txt"
OUTPUT_PATH = "/workspace/output/processed.txt"
VOCAB_SIZE = 30000
MASK_PERCENTAGE = 0.15

def load_vocab():
    """Load vocabulary from file and create token to ID mapping."""
    vocab = {}
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            if token:
                vocab[token] = idx
    return vocab

def tokenize(text):
    """Simple whitespace tokenization."""
    return text.strip().split()

def tokens_to_ids(tokens, vocab):
    """
    Convert tokens to IDs using vocabulary.
    BUG 1: Token ID overflow - uses hash function that can exceed vocab size.
    """
    token_ids = []
    for token in tokens:
        if token in vocab:
            # BUG: Using hash with large modulo that exceeds vocab size
            token_id = abs(hash(token)) % 50000
            token_ids.append(token_id)
        else:
            # Unknown token - also buggy
            token_id = abs(hash(token)) % 50000
            token_ids.append(token_id)
    return token_ids

def apply_masking(token_ids):
    """
    Apply masking to token IDs for MLM training.
    BUG 2: Masks consecutive positions from start instead of random 15%.
    """
    num_tokens = len(token_ids)
    num_to_mask = max(1, int(num_tokens * MASK_PERCENTAGE))
    
    # BUG: Always mask consecutive positions starting from 0
    masked_positions = list(range(num_to_mask))
    
    return masked_positions

def process_corpus():
    """Process the entire corpus and generate training data."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Load vocabulary
    vocab = load_vocab()
    
    # Process corpus
    with open(CORPUS_PATH, 'r', encoding='utf-8') as infile:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                # Tokenize
                tokens = tokenize(line)
                if len(tokens) == 0:
                    continue
                
                # Convert to IDs (with overflow bug)
                token_ids = tokens_to_ids(tokens, vocab)
                
                # Apply masking (with consecutive position bug)
                masked_positions = apply_masking(token_ids)
                
                # BUG 3: Write in JSON format instead of required format
                output_obj = {
                    "tokens": token_ids,
                    "masked_positions": masked_positions
                }
                outfile.write(json.dumps(output_obj) + '\n')

def main():
    """Main entry point."""
    print("Starting preprocessing...")
    print(f"Loading vocabulary from {VOCAB_PATH}")
    print(f"Processing corpus from {CORPUS_PATH}")
    print(f"Output will be written to {OUTPUT_PATH}")
    
    process_corpus()
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()