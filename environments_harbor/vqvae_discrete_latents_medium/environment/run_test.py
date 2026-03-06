#!/usr/bin/env python3

import numpy as np
from quantizer import VectorQuantizer

def main():
    print("Loading test data...")
    test_data = np.load('/workspace/test_data.npy')
    print(f"Test data shape: {test_data.shape}")
    
    # Create VectorQuantizer with 64 codebook entries and vector dimension 8
    print("\nInitializing VectorQuantizer...")
    quantizer = VectorQuantizer(num_codes=64, vector_dim=8)
    
    # Train the quantizer on the test data
    print("Training quantizer on test data...")
    quantizer.fit(test_data)
    
    # Run quantization
    print("\nRunning quantization...")
    quantized_vectors, indices = quantizer.quantize(test_data)
    
    print(f"Quantized vectors shape: {quantized_vectors.shape}")
    print(f"Indices shape: {indices.shape}")
    
    # Validation checks
    all_checks_passed = True
    
    # Check 1: All quantized vectors exactly match entries in the codebook
    print("\n=== Check 1: Verifying quantized vectors match codebook entries ===")
    codebook = quantizer.codebook
    vectors_match_codebook = True
    for i, vec in enumerate(quantized_vectors):
        # Check if this vector exists in the codebook
        matches = np.all(np.isclose(codebook, vec, atol=1e-9), axis=1)
        if not np.any(matches):
            print(f"FAIL: Vector at index {i} does not match any codebook entry")
            vectors_match_codebook = False
            all_checks_passed = False
            if i >= 5:  # Only show first few errors
                print("(More errors omitted...)")
                break
    
    if vectors_match_codebook:
        print("PASS: All quantized vectors match codebook entries")
    
    # Check 2: All indices are valid
    print("\n=== Check 2: Verifying indices are valid ===")
    num_codes = codebook.shape[0]
    indices_valid = np.all((indices >= 0) & (indices < num_codes))
    if indices_valid:
        print(f"PASS: All indices are in valid range [0, {num_codes-1}]")
    else:
        print(f"FAIL: Some indices are out of valid range [0, {num_codes-1}]")
        print(f"Min index: {np.min(indices)}, Max index: {np.max(indices)}")
        all_checks_passed = False
    
    # Check 3: Indices correctly map to quantized vectors
    print("\n=== Check 3: Verifying indices map correctly to quantized vectors ===")
    if indices_valid:
        reconstructed_from_indices = codebook[indices]
        indices_map_correctly = np.allclose(reconstructed_from_indices, quantized_vectors, atol=1e-9)
        if indices_map_correctly:
            print("PASS: Indices correctly map to quantized vectors")
        else:
            print("FAIL: Indices do not correctly map to quantized vectors")
            # Show a few mismatches
            mismatches = ~np.all(np.isclose(reconstructed_from_indices, quantized_vectors, atol=1e-9), axis=1)
            if np.any(mismatches):
                mismatch_indices = np.where(mismatches)[0][:5]
                for idx in mismatch_indices:
                    print(f"  Mismatch at position {idx}:")
                    print(f"    Index: {indices[idx]}")
                    print(f"    Expected (quantized): {quantized_vectors[idx]}")
                    print(f"    Got (from codebook): {reconstructed_from_indices[idx]}")
            all_checks_passed = False
    else:
        print("SKIP: Cannot verify index mapping due to invalid indices")
        all_checks_passed = False
    
    # Check 4: Calculate reconstruction error
    print("\n=== Check 4: Calculating reconstruction error ===")
    errors = np.linalg.norm(test_data - quantized_vectors, axis=1)
    avg_error = np.mean(errors)
    print(f"Average reconstruction error: {avg_error:.6f}")
    print(f"Min error: {np.min(errors):.6f}, Max error: {np.max(errors):.6f}")
    
    error_in_range = 0.1 <= avg_error <= 2.0
    if error_in_range:
        print(f"PASS: Error is in reasonable range [0.1, 2.0]")
    else:
        print(f"FAIL: Error {avg_error:.6f} is outside reasonable range [0.1, 2.0]")
        all_checks_passed = False
    
    # Check 5: Count unique codebook entries used
    print("\n=== Check 5: Counting unique codebook entries used ===")
    unique_codes = len(np.unique(indices))
    print(f"Unique codebook entries used: {unique_codes} out of {num_codes}")
    
    sufficient_diversity = unique_codes >= 20
    if sufficient_diversity:
        print(f"PASS: At least 20 unique codes used")
    else:
        print(f"FAIL: Only {unique_codes} unique codes used (need at least 20)")
        all_checks_passed = False
    
    # Determine final status
    print("\n" + "="*60)
    if all_checks_passed:
        status = "PASS"
        print("OVERALL STATUS: PASS - All checks passed!")
    else:
        status = "FAIL"
        print("OVERALL STATUS: FAIL - Some checks failed")
    print("="*60)
    
    # Write results to solution.txt
    print(f"\nWriting results to /workspace/solution.txt...")
    with open('/workspace/solution.txt', 'w') as f:
        f.write(f"STATUS={status}\n")
        f.write(f"AVG_ERROR={avg_error:.6f}\n")
        f.write(f"UNIQUE_CODES={unique_codes}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()