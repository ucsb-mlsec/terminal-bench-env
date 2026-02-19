#!/usr/bin/env python3

import json
import sys
import os
import re

def load_json_file(filepath):
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {filepath}: {e}")
        return None

def test_markers_present(vocab):
    """Test that all five special markers are present in vocabulary."""
    markers = ['[PII]', '[FINANCIAL]', '[MEDICAL]', '[CONFIDENTIAL]', '[URGENT]']
    missing = []
    
    for marker in markers:
        if marker not in vocab:
            missing.append(marker)
    
    if missing:
        print(f"FAIL: Missing markers in vocabulary: {missing}")
        return False
    else:
        print("PASS: All 5 special markers present in vocabulary")
        return True

def test_unique_ids(vocab):
    """Test that each marker has a unique integer ID."""
    markers = ['[PII]', '[FINANCIAL]', '[MEDICAL]', '[CONFIDENTIAL]', '[URGENT]']
    marker_ids = []
    
    for marker in markers:
        if marker in vocab:
            marker_id = vocab[marker]
            if not isinstance(marker_id, int):
                print(f"FAIL: Marker {marker} has non-integer ID: {marker_id}")
                return False
            marker_ids.append(marker_id)
    
    if len(marker_ids) != len(set(marker_ids)):
        print(f"FAIL: Marker IDs are not unique: {marker_ids}")
        return False
    
    print(f"PASS: All markers have unique integer IDs: {marker_ids}")
    return True

def test_vocab_size_increase(original_vocab, updated_vocab):
    """Test that vocabulary size increased by exactly 5."""
    original_size = len(original_vocab)
    updated_size = len(updated_vocab)
    increase = updated_size - original_size
    
    if increase == 5:
        print(f"PASS: Vocabulary size increased by exactly 5 (from {original_size} to {updated_size})")
        return True
    else:
        print(f"FAIL: Vocabulary size increased by {increase}, expected 5 (original: {original_size}, updated: {updated_size})")
        return False

def test_original_tokens_retained(original_vocab, updated_vocab):
    """Test that original tokens retained their IDs."""
    mismatched = []
    
    for token, orig_id in original_vocab.items():
        if token in updated_vocab:
            if updated_vocab[token] != orig_id:
                mismatched.append((token, orig_id, updated_vocab[token]))
        else:
            mismatched.append((token, orig_id, 'MISSING'))
    
    if mismatched:
        print(f"FAIL: Original tokens have different IDs or are missing:")
        for token, orig_id, new_id in mismatched[:5]:  # Show first 5
            print(f"  {token}: {orig_id} -> {new_id}")
        return False
    else:
        print(f"PASS: All {len(original_vocab)} original tokens retained their IDs")
        return True

def tokenize_text(text, vocab):
    """
    Tokenize text using longest-match-first strategy.
    Returns list of tokens or None if tokenization fails.
    """
    tokens = []
    i = 0
    
    # Sort vocabulary by length (longest first) to implement longest-match
    sorted_vocab_tokens = sorted(vocab.keys(), key=len, reverse=True)
    
    while i < len(text):
        matched = False
        
        # Try to match the longest token first
        for vocab_token in sorted_vocab_tokens:
            if text[i:i+len(vocab_token)] == vocab_token:
                tokens.append(vocab_token)
                i += len(vocab_token)
                matched = True
                break
        
        if not matched:
            # Character not in vocabulary - could be whitespace or unknown
            if text[i].isspace():
                i += 1
            else:
                # Unknown character
                tokens.append(f"[UNK:{text[i]}]")
                i += 1
    
    return tokens

def test_markers_not_split(updated_vocab, sample_file):
    """Test that markers are not split during tokenization."""
    if not os.path.exists(sample_file):
        print(f"FAIL: Sample file not found: {sample_file}")
        return False
    
    try:
        with open(sample_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"FAIL: Could not read sample file: {e}")
        return False
    
    markers = ['[PII]', '[FINANCIAL]', '[MEDICAL]', '[CONFIDENTIAL]', '[URGENT]']
    
    # Tokenize the content
    tokens = tokenize_text(content, updated_vocab)
    
    # Check if any marker was split
    split_markers = []
    for marker in markers:
        if marker in content and marker not in tokens:
            # Check if parts of the marker appear as separate tokens
            marker_parts = ['[', marker[1:-1], ']']
            if all(part in tokens for part in marker_parts):
                split_markers.append(marker)
    
    if split_markers:
        print(f"FAIL: Markers were split during tokenization: {split_markers}")
        return False
    
    # Count occurrences of complete markers in original text vs tokens
    markers_found = 0
    for marker in markers:
        text_count = content.count(marker)
        token_count = tokens.count(marker)
        markers_found += token_count
        if text_count > 0 and token_count != text_count:
            print(f"FAIL: Marker {marker} appears {text_count} times in text but {token_count} times in tokens")
            return False
    
    print(f"PASS: All markers tokenized correctly ({markers_found} marker instances found in sample tickets)")
    return True

def main():
    """Run all tests and print summary."""
    print("=" * 70)
    print("TOKENIZER CONFIGURATION TEST SUITE")
    print("=" * 70)
    print()
    
    # File paths
    original_config_path = '/workspace/tokenizer_config.json'
    updated_config_path = '/workspace/updated_tokenizer_config.json'
    sample_tickets_path = '/workspace/sample_tickets.txt'
    
    # Load configurations
    print("Loading configuration files...")
    original_config = load_json_file(original_config_path)
    updated_config = load_json_file(updated_config_path)
    
    if original_config is None or updated_config is None:
        print("\nFATAL: Could not load required configuration files")
        sys.exit(1)
    
    original_vocab = original_config.get('vocab', {})
    updated_vocab = updated_config.get('vocab', {})
    
    if not original_vocab:
        print("ERROR: Original config has no vocab")
        sys.exit(1)
    
    if not updated_vocab:
        print("ERROR: Updated config has no vocab")
        sys.exit(1)
    
    print(f"Original vocabulary size: {len(original_vocab)}")
    print(f"Updated vocabulary size: {len(updated_vocab)}")
    print()
    
    # Run tests
    results = []
    
    print("Test 1: Checking if all special markers are present...")
    results.append(test_markers_present(updated_vocab))
    print()
    
    print("Test 2: Checking if markers have unique integer IDs...")
    results.append(test_unique_ids(updated_vocab))
    print()
    
    print("Test 3: Checking vocabulary size increase...")
    results.append(test_vocab_size_increase(original_vocab, updated_vocab))
    print()
    
    print("Test 4: Checking if original tokens retained their IDs...")
    results.append(test_original_tokens_retained(original_vocab, updated_vocab))
    print()
    
    print("Test 5: Checking if markers are not split during tokenization...")
    results.append(test_markers_not_split(updated_vocab, sample_tickets_path))
    print()
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print()
    
    if all(results):
        print("✓ ALL TESTS PASSED - Tokenizer configuration is correct!")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED - Please review the tokenizer configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()