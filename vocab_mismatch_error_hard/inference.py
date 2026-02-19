#!/usr/bin/env python3

import json
import sys
import os

def load_model_config(config_path):
    """Load model configuration and extract expected vocabulary size."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'vocab_size' not in config:
            raise KeyError("'vocab_size' not found in model configuration")
        
        expected_vocab_size = config['vocab_size']
        print(f"Model expects vocabulary size: {expected_vocab_size}")
        return expected_vocab_size
    
    except FileNotFoundError:
        print(f"Error: Model configuration file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse model configuration JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        sys.exit(1)

def load_tokenizer_vocab(vocab_path):
    """Load tokenizer vocabulary and count actual tokens."""
    try:
        with open(vocab_path, 'r') as f:
            lines = f.readlines()
        
        # Count non-empty lines
        actual_vocab_size = len([line for line in lines if line.strip()])
        print(f"Tokenizer vocabulary size: {actual_vocab_size}")
        return actual_vocab_size
    
    except FileNotFoundError:
        print(f"Error: Tokenizer vocabulary file not found at {vocab_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tokenizer vocabulary: {e}")
        sys.exit(1)

def validate_vocabulary(expected_size, actual_size):
    """Validate that vocabulary sizes match."""
    if expected_size != actual_size:
        raise ValueError(
            f"Vocabulary size mismatch. Expected: {expected_size}, Got: {actual_size}"
        )
    print("Vocabulary validation passed!")

def load_test_input(input_path):
    """Load test input text."""
    try:
        with open(input_path, 'r') as f:
            text = f.read().strip()
        
        if not text:
            raise ValueError("Test input file is empty")
        
        print(f"Loaded test input: {text[:50]}..." if len(text) > 50 else f"Loaded test input: {text}")
        return text
    
    except FileNotFoundError:
        print(f"Error: Test input file not found at {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading test input: {e}")
        sys.exit(1)

def run_inference(text):
    """Perform mock inference on input text."""
    # Simulated inference since actual model isn't present
    print("\n=== Running Inference ===")
    print(f"Input text: {text}")
    
    # Mock tokenization
    tokens = text.lower().split()
    print(f"Tokenized: {tokens}")
    
    # Mock classification result
    classes = ['positive', 'negative', 'neutral']
    mock_scores = [0.7, 0.2, 0.1]
    
    print("\nClassification Results:")
    for cls, score in zip(classes, mock_scores):
        print(f"  {cls}: {score:.2f}")
    
    predicted_class = classes[0]
    print(f"\nPredicted class: {predicted_class}")
    
    return predicted_class

def main():
    print("=== Text Classification Inference Pipeline ===\n")
    
    # Define paths
    config_path = '/workspace/model_config.json'
    vocab_path = '/workspace/tokenizer_vocab.txt'
    input_path = '/workspace/test_input.txt'
    
    # Step 1: Load model configuration
    print("Step 1: Loading model configuration...")
    expected_vocab_size = load_model_config(config_path)
    
    # Step 2: Load tokenizer vocabulary
    print("\nStep 2: Loading tokenizer vocabulary...")
    actual_vocab_size = load_tokenizer_vocab(vocab_path)
    
    # Step 3: Validate vocabulary compatibility
    print("\nStep 3: Validating vocabulary compatibility...")
    try:
        validate_vocabulary(expected_vocab_size, actual_vocab_size)
    except ValueError as e:
        print(f"\nValidation Failed: {e}")
        sys.exit(1)
    
    # Step 4: Load test input
    print("\nStep 4: Loading test input...")
    test_text = load_test_input(input_path)
    
    # Step 5: Run inference
    print("\nStep 5: Running inference...")
    result = run_inference(test_text)
    
    print("\n=== Inference Completed Successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())