#!/usr/bin/env python3

import json
import numpy as np
import sys
import os

def validate_codebook(codebook_path, sample_tokens_path):
    """Validate a token codebook against requirements."""
    
    # Load codebook
    try:
        with open(codebook_path, 'r') as f:
            codebook = json.load(f)
    except FileNotFoundError:
        print(f"Error: Codebook file not found at {codebook_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in codebook file: {e}")
        return False
    
    # Load sample tokens
    try:
        sample_tokens = np.load(sample_tokens_path)
    except FileNotFoundError:
        print(f"Error: Sample tokens file not found at {sample_tokens_path}")
        return False
    except Exception as e:
        print(f"Error loading sample tokens: {e}")
        return False
    
    print("=" * 60)
    print("CODEBOOK VALIDATION REPORT")
    print("=" * 60)
    
    # Check total entries
    total_entries = len(codebook)
    print(f"\nTotal codebook entries: {total_entries}")
    
    # Validate each entry
    invalid_entries = {}
    valid_count = 0
    
    # Check all token IDs 0-255
    for token_id in range(256):
        token_key = str(token_id)
        
        if token_key not in codebook:
            invalid_entries[token_id] = f"Token {token_id}: missing from codebook"
            continue
        
        entry = codebook[token_key]
        
        # Check if entry has vector field
        if "vector" not in entry:
            invalid_entries[token_id] = f"Token {token_id}: missing vector field"
            continue
        
        vector = entry["vector"]
        
        # Check if vector is a list
        if not isinstance(vector, list):
            invalid_entries[token_id] = f"Token {token_id}: vector is not a list (type: {type(vector).__name__})"
            continue
        
        # Check vector length
        if len(vector) != 8:
            invalid_entries[token_id] = f"Token {token_id}: vector has {len(vector)} elements instead of 8"
            continue
        
        # Check all elements are numeric
        non_numeric = []
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)) or val is None:
                non_numeric.append(i)
        
        if non_numeric:
            invalid_entries[token_id] = f"Token {token_id}: vector contains non-numeric values at positions {non_numeric}"
            continue
        
        # If we got here, entry is valid
        valid_count += 1
    
    invalid_count = len(invalid_entries)
    
    print(f"Valid entries: {valid_count}")
    print(f"Invalid entries: {invalid_count}")
    
    # Print invalid entries
    if invalid_entries:
        print("\nINVALID ENTRIES:")
        print("-" * 60)
        for token_id in sorted(invalid_entries.keys()):
            print(f"  {invalid_entries[token_id]}")
    
    # Check sample tokens
    unique_tokens = np.unique(sample_tokens)
    print(f"\nSample image analysis:")
    print(f"  Unique token IDs in sample: {len(unique_tokens)}")
    print(f"  Token ID range: {unique_tokens.min()} to {unique_tokens.max()}")
    
    # Check which sample tokens reference invalid entries
    invalid_sample_tokens = []
    for token_id in unique_tokens:
        if token_id in invalid_entries:
            invalid_sample_tokens.append(token_id)
    
    if invalid_sample_tokens:
        print(f"\nWARNING: {len(invalid_sample_tokens)} token IDs from sample image reference invalid codebook entries:")
        print("-" * 60)
        for token_id in sorted(invalid_sample_tokens):
            print(f"  {invalid_entries[token_id]}")
    else:
        print("\n✓ All token IDs from sample image reference valid codebook entries")
    
    print("\n" + "=" * 60)
    
    # Overall validation result
    if invalid_count == 0:
        print("RESULT: Codebook is VALID ✓")
        return True
    else:
        print(f"RESULT: Codebook is INVALID - {invalid_count} entries need repair ✗")
        return False

if __name__ == "__main__":
    codebook_path = "/workspace/codebook.json"
    sample_tokens_path = "/workspace/sample_tokens.npy"
    
    is_valid = validate_codebook(codebook_path, sample_tokens_path)
    sys.exit(0 if is_valid else 1)