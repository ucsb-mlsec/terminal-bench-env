#!/usr/bin/env python3

import numpy as np
import math
import sys

def test_value_range_64_10():
    """Test that all values are in [-1, 1] for d_model=64, seq_len=10"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=64, max_seq_len=10)
    encoding = pe.get_encoding(10)
    return np.all(encoding >= -1.0) and np.all(encoding <= 1.0)

def test_value_range_128_50():
    """Test that all values are in [-1, 1] for d_model=128, seq_len=50"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=128, max_seq_len=50)
    encoding = pe.get_encoding(50)
    return np.all(encoding >= -1.0) and np.all(encoding <= 1.0)

def test_value_range_256_100():
    """Test that all values are in [-1, 1] for d_model=256, seq_len=100"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=256, max_seq_len=100)
    encoding = pe.get_encoding(100)
    return np.all(encoding >= -1.0) and np.all(encoding <= 1.0)

def test_value_range_512_500():
    """Test that all values are in [-1, 1] for d_model=512, seq_len=500"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=512, max_seq_len=500)
    encoding = pe.get_encoding(500)
    return np.all(encoding >= -1.0) and np.all(encoding <= 1.0)

def test_shape_64_10():
    """Test output shape for d_model=64, seq_len=10"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=64, max_seq_len=10)
    encoding = pe.get_encoding(10)
    return encoding.shape == (10, 64)

def test_shape_128_50():
    """Test output shape for d_model=128, seq_len=50"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=128, max_seq_len=50)
    encoding = pe.get_encoding(50)
    return encoding.shape == (50, 128)

def test_shape_256_100():
    """Test output shape for d_model=256, seq_len=100"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=256, max_seq_len=100)
    encoding = pe.get_encoding(100)
    return encoding.shape == (100, 256)

def test_shape_512_500():
    """Test output shape for d_model=512, seq_len=500"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=512, max_seq_len=500)
    encoding = pe.get_encoding(500)
    return encoding.shape == (500, 512)

def test_position_zero_even_dims():
    """Test that pos=0 gives sin(0)=0 for even dimensions"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=64, max_seq_len=10)
    encoding = pe.get_encoding(10)
    even_dims = encoding[0, 0::2]
    return np.allclose(even_dims, 0.0, atol=1e-6)

def test_position_zero_odd_dims():
    """Test that pos=0 gives cos(0)=1 for odd dimensions"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=64, max_seq_len=10)
    encoding = pe.get_encoding(10)
    odd_dims = encoding[0, 1::2]
    return np.allclose(odd_dims, 1.0, atol=1e-6)

def test_first_dimension_sine():
    """Test first dimension (dim=0) uses sine function"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 10
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    expected = np.array([np.sin(pos / 10000**(0/d_model)) for pos in range(seq_len)])
    return np.allclose(encoding[:, 0], expected, rtol=1e-5)

def test_second_dimension_cosine():
    """Test second dimension (dim=1) uses cosine function"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 10
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    expected = np.array([np.cos(pos / 10000**(0/d_model)) for pos in range(seq_len)])
    return np.allclose(encoding[:, 1], expected, rtol=1e-5)

def test_sine_formula_dim_0():
    """Test sine formula for dimension 0"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 20
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    i = 0
    expected = np.array([np.sin(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    return np.allclose(encoding[:, 2*i], expected, rtol=1e-5)

def test_cosine_formula_dim_1():
    """Test cosine formula for dimension 1"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 20
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    i = 0
    expected = np.array([np.cos(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    return np.allclose(encoding[:, 2*i+1], expected, rtol=1e-5)

def test_sine_formula_dim_10():
    """Test sine formula for dimension 10"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 20
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    i = 5
    expected = np.array([np.sin(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    return np.allclose(encoding[:, 2*i], expected, rtol=1e-5)

def test_cosine_formula_dim_11():
    """Test cosine formula for dimension 11"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 20
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    i = 5
    expected = np.array([np.cos(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    return np.allclose(encoding[:, 2*i+1], expected, rtol=1e-5)

def test_frequency_scaling():
    """Test that frequencies decrease geometrically across dimensions"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    pe = PositionalEncoding(d_model=d_model, max_seq_len=100)
    encoding = pe.get_encoding(100)
    
    # Check that later dimensions have lower frequencies (longer periods)
    # by comparing variance at different dimension pairs
    for i in range(3):
        var_early = np.var(encoding[:, 2*i])
        var_late = np.var(encoding[:, 2*(i+1)])
        # Later dimensions should have less variance for same sequence length
        if var_late > var_early:
            return False
    return True

def test_wavelength_increases():
    """Test that wavelengths increase geometrically"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    
    # Calculate expected wavelengths
    wavelengths = []
    for i in range(d_model // 2):
        wavelength = 2 * np.pi * (10000 ** (2*i/d_model))
        wavelengths.append(wavelength)
    
    # Check they increase
    for i in range(len(wavelengths)-1):
        if wavelengths[i+1] <= wavelengths[i]:
            return False
    return True

def test_even_dimensions_all_sine():
    """Test all even dimensions use sine"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 50
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    for i in range(d_model // 2):
        dim_idx = 2 * i
        expected = np.array([np.sin(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
        if not np.allclose(encoding[:, dim_idx], expected, rtol=1e-5):
            return False
    return True

def test_odd_dimensions_all_cosine():
    """Test all odd dimensions use cosine"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 50
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    for i in range(d_model // 2):
        dim_idx = 2 * i + 1
        expected = np.array([np.cos(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
        if not np.allclose(encoding[:, dim_idx], expected, rtol=1e-5):
            return False
    return True

def test_small_dmodel_4():
    """Test with small d_model=4"""
    from positional_encoding import PositionalEncoding
    d_model = 4
    seq_len = 10
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    return encoding.shape == (seq_len, d_model) and np.all(encoding >= -1) and np.all(encoding <= 1)

def test_small_dmodel_8():
    """Test with small d_model=8"""
    from positional_encoding import PositionalEncoding
    d_model = 8
    seq_len = 10
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    return encoding.shape == (seq_len, d_model) and np.all(encoding >= -1) and np.all(encoding <= 1)

def test_large_seq_len_1000():
    """Test with large sequence length"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=64, max_seq_len=1000)
    encoding = pe.get_encoding(1000)
    return encoding.shape == (1000, 64)

def test_position_independence():
    """Test that different positions have different encodings"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=64, max_seq_len=100)
    encoding = pe.get_encoding(100)
    
    # Check that position 0 and position 1 are different
    return not np.allclose(encoding[0], encoding[1])

def test_dimension_pair_orthogonality():
    """Test sine/cosine pairs have correct relationship"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 10
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # For each dimension pair, sin^2 + cos^2 should equal 1
    for i in range(d_model // 2):
        sin_vals = encoding[:, 2*i]
        cos_vals = encoding[:, 2*i+1]
        sum_squares = sin_vals**2 + cos_vals**2
        if not np.allclose(sum_squares, 1.0, rtol=1e-5):
            return False
    return True

def test_specific_value_pos1_dim0():
    """Test specific value at position 1, dimension 0"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    pe = PositionalEncoding(d_model=d_model, max_seq_len=10)
    encoding = pe.get_encoding(10)
    
    expected = np.sin(1 / (10000 ** (0/d_model)))
    return np.allclose(encoding[1, 0], expected, rtol=1e-5)

def test_specific_value_pos5_dim2():
    """Test specific value at position 5, dimension 2"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    pe = PositionalEncoding(d_model=d_model, max_seq_len=10)
    encoding = pe.get_encoding(10)
    
    i = 1
    expected = np.sin(5 / (10000 ** (2*i/d_model)))
    return np.allclose(encoding[5, 2], expected, rtol=1e-5)

def test_specific_value_pos3_dim3():
    """Test specific value at position 3, dimension 3"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    pe = PositionalEncoding(d_model=d_model, max_seq_len=10)
    encoding = pe.get_encoding(10)
    
    i = 1
    expected = np.cos(3 / (10000 ** (2*i/d_model)))
    return np.allclose(encoding[3, 3], expected, rtol=1e-5)

def test_periodicity_low_frequency():
    """Test periodicity of low frequency components"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 500
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # First dimension should complete cycles within sequence
    # Check that it's periodic-like
    vals = encoding[:, 0]
    # Should have both positive and negative values
    return np.any(vals > 0) and np.any(vals < 0)

def test_last_dimensions_low_frequency():
    """Test that last dimensions have very low frequency"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 100
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # Last dimension should change very slowly
    last_dim = encoding[:, -2]  # Last sine dimension
    # Variance should be relatively small for short sequences
    return np.var(last_dim) < 0.1

def test_dmodel_256_consistency():
    """Test d_model=256 produces consistent results"""
    from positional_encoding import PositionalEncoding
    d_model = 256
    seq_len = 50
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # Verify basic formula on sample dimensions
    i = 10
    expected_sin = np.array([np.sin(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    expected_cos = np.array([np.cos(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    
    return (np.allclose(encoding[:, 2*i], expected_sin, rtol=1e-5) and 
            np.allclose(encoding[:, 2*i+1], expected_cos, rtol=1e-5))

def test_dmodel_512_consistency():
    """Test d_model=512 produces consistent results"""
    from positional_encoding import PositionalEncoding
    d_model = 512
    seq_len = 50
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # Verify basic formula on sample dimensions
    i = 20
    expected_sin = np.array([np.sin(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    expected_cos = np.array([np.cos(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    
    return (np.allclose(encoding[:, 2*i], expected_sin, rtol=1e-5) and 
            np.allclose(encoding[:, 2*i+1], expected_cos, rtol=1e-5))

def test_no_nan_values():
    """Test that no NaN values are present"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=128, max_seq_len=100)
    encoding = pe.get_encoding(100)
    return not np.any(np.isnan(encoding))

def test_no_inf_values():
    """Test that no infinite values are present"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=128, max_seq_len=100)
    encoding = pe.get_encoding(100)
    return not np.any(np.isinf(encoding))

def test_deterministic_output():
    """Test that same inputs give same outputs"""
    from positional_encoding import PositionalEncoding
    pe1 = PositionalEncoding(d_model=64, max_seq_len=50)
    pe2 = PositionalEncoding(d_model=64, max_seq_len=50)
    enc1 = pe1.get_encoding(50)
    enc2 = pe2.get_encoding(50)
    return np.allclose(enc1, enc2)

def test_middle_dimensions():
    """Test middle dimensions follow correct formula"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 30
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # Test middle dimension pair
    i = d_model // 4
    expected_sin = np.array([np.sin(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    expected_cos = np.array([np.cos(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    
    return (np.allclose(encoding[:, 2*i], expected_sin, rtol=1e-5) and 
            np.allclose(encoding[:, 2*i+1], expected_cos, rtol=1e-5))

def test_last_dimension_pair():
    """Test last dimension pair follows correct formula"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 20
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # Test last dimension pair
    i = (d_model // 2) - 1
    expected_sin = np.array([np.sin(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    expected_cos = np.array([np.cos(pos / (10000 ** (2*i/d_model))) for pos in range(seq_len)])
    
    return (np.allclose(encoding[:, 2*i], expected_sin, rtol=1e-5) and 
            np.allclose(encoding[:, 2*i+1], expected_cos, rtol=1e-5))

def test_position_10_all_dims():
    """Test position 10 across all dimensions"""
    from positional_encoding import PositionalEncoding
    d_model = 32
    seq_len = 20
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    pos = 10
    for i in range(d_model // 2):
        expected_sin = np.sin(pos / (10000 ** (2*i/d_model)))
        expected_cos = np.cos(pos / (10000 ** (2*i/d_model)))
        if not np.allclose(encoding[pos, 2*i], expected_sin, rtol=1e-5):
            return False
        if not np.allclose(encoding[pos, 2*i+1], expected_cos, rtol=1e-5):
            return False
    return True

def test_full_encoding_64_20():
    """Full validation for d_model=64, seq_len=20"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 20
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    for pos in range(seq_len):
        for i in range(d_model // 2):
            expected_sin = np.sin(pos / (10000 ** (2*i/d_model)))
            expected_cos = np.cos(pos / (10000 ** (2*i/d_model)))
            if not np.allclose(encoding[pos, 2*i], expected_sin, rtol=1e-5):
                return False
            if not np.allclose(encoding[pos, 2*i+1], expected_cos, rtol=1e-5):
                return False
    return True

def test_full_encoding_128_30():
    """Full validation for d_model=128, seq_len=30"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 30
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    for pos in range(seq_len):
        for i in range(d_model // 2):
            expected_sin = np.sin(pos / (10000 ** (2*i/d_model)))
            expected_cos = np.cos(pos / (10000 ** (2*i/d_model)))
            if not np.allclose(encoding[pos, 2*i], expected_sin, rtol=1e-5):
                return False
            if not np.allclose(encoding[pos, 2*i+1], expected_cos, rtol=1e-5):
                return False
    return True

def test_alternating_pattern():
    """Test that dimensions alternate between sine and cosine"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    seq_len = 10
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # Check alternating pattern at pos=0
    # Even dims should be 0 (sine), odd dims should be 1 (cosine)
    for dim in range(d_model):
        if dim % 2 == 0:
            if not np.allclose(encoding[0, dim], 0.0, atol=1e-6):
                return False
        else:
            if not np.allclose(encoding[0, dim], 1.0, atol=1e-6):
                return False
    return True

def test_geometric_progression():
    """Test geometric progression of frequencies"""
    from positional_encoding import PositionalEncoding
    d_model = 64
    
    # Calculate ratios between consecutive dimension pairs
    ratios = []
    for i in range(1, d_model // 2):
        freq_prev = 1 / (10000 ** (2*(i-1)/d_model))
        freq_curr = 1 / (10000 ** (2*i/d_model))
        ratio = freq_prev / freq_curr
        ratios.append(ratio)
    
    # All ratios should be approximately equal (geometric progression)
    if len(ratios) > 1:
        for i in range(len(ratios)-1):
            if not np.allclose(ratios[i], ratios[i+1], rtol=0.01):
                return False
    return True

def test_max_frequency_first_dim():
    """Test that first dimension has highest frequency"""
    from positional_encoding import PositionalEncoding
    d_model = 128
    seq_len = 100
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    encoding = pe.get_encoding(seq_len)
    
    # First dimension should have most zero-crossings
    first_dim = encoding[:, 0]
    last_dim = encoding[:, -2]
    
    # Count zero crossings
    first_crossings = np.sum(np.diff(np.sign(first_dim)) != 0)
    last_crossings = np.sum(np.diff(np.sign(last_dim)) != 0)
    
    return first_crossings >= last_crossings

def test_formula_consistency_multiple_configs():
    """Test formula consistency across multiple configurations"""
    from positional_encoding import PositionalEncoding
    
    configs = [(64, 20), (128, 30), (256, 40)]
    
    for d_model, seq_len in configs:
        pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len)
        encoding = pe.get_encoding(seq_len)
        
        # Sample a few positions and dimensions
        for pos in [0, 5, 10]:
            if pos >= seq_len:
                continue
            for i in [0, d_model//8, d_model//4]:
                if 2*i+1 >= d_model:
                    continue
                expected_sin = np.sin(pos / (10000 ** (2*i/d_model)))
                expected_cos = np.cos(pos / (10000 ** (2*i/d_model)))
                if not np.allclose(encoding[pos, 2*i], expected_sin, rtol=1e-5):
                    return False
                if not np.allclose(encoding[pos, 2*i+1], expected_cos, rtol=1e-5):
                    return False
    return True

def test_seq_len_boundary():
    """Test at sequence length boundary"""
    from positional_encoding import PositionalEncoding
    max_seq_len = 100
    pe = PositionalEncoding(d_model=64, max_seq_len=max_seq_len)
    encoding = pe.get_encoding(max_seq_len)
    
    return encoding.shape[0] == max_seq_len

def test_numerical_stability():
    """Test numerical stability with large positions"""
    from positional_encoding import PositionalEncoding
    pe = PositionalEncoding(d_model=512, max_seq_len=1000)
    encoding = pe.get_encoding(1000)
    
    # Check that values remain bounded
    return np.all(np.abs(encoding) <= 1.0 + 1e-6)

def run_all_tests():
    """Run all validation tests and return results"""
    tests = [
        test_value_range_64_10,
        test_value_range_128_50,
        test_value_range_256_100,
        test_value_range_512_500,
        test_shape_64_10,
        test_shape_128_50,
        test_shape_256_100,
        test_shape_512_500,
        test_position_zero_even_dims,
        test_position_zero_odd_dims,
        test_first_dimension_sine,
        test_second_dimension_cosine,
        test_sine_formula_dim_0,
        test_cosine_formula_dim_1,
        test_sine_formula_dim_10,
        test_cosine_formula_dim_11,
        test_frequency_scaling,
        test_wavelength_increases,
        test_even_dimensions_all_sine,
        test_odd_dimensions_all_cosine,
        test_small_dmodel_4,
        test_small_dmodel_8,
        test_large_seq_len_1000,
        test_position_independence,
        test_dimension_pair_orthogonality,
        test_specific_value_pos1_dim0,
        test_specific_value_pos5_dim2,
        test_specific_value_pos3_dim3,
        test_periodicity_low_frequency,
        test_last_dimensions_low_frequency,
        test_dmodel_256_consistency,
        test_dmodel_512_consistency,
        test_no_nan_values,
        test_no_inf_values,
        test_deterministic_output,
        test_middle_dimensions,
        test_last_dimension_pair,
        test_position_10_all_dims,
        test_full_encoding_64_20,
        test_full_encoding_128_30,
        test_alternating_pattern,
        test_geometric_progression,
        test_max_frequency_first_dim,
        test_formula_consistency_multiple_configs,
        test_seq_len_boundary,
        test_numerical_stability,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
    
    return {
        'tests_passed': passed,
        'tests_failed': failed,
        'implementation_correct': failed == 0
    }

if __name__ == '__main__':
    results = run_all_tests()
    print(f"Tests passed: {results['tests_passed']}")
    print(f"Tests failed: {results['tests_failed']}")
    print(f"Implementation correct: {results['implementation_correct']}")