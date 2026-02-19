#!/usr/bin/env python3

import os
import sys
import pytest
import shutil
import subprocess

# First, let's create the crypto_app.py file
crypto_app_code = '''#!/usr/bin/env python3

import requests

API_BASE_URL = "https://api.coinbase.com/v2/prices"

def get_crypto_price(crypto_code):
    """Fetch the current price for a cryptocurrency."""
    url = f"{API_BASE_URL}/{crypto_code}/spot"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('data', {})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching price for {crypto_code}: {e}")
        return None

def get_multiple_prices(crypto_codes):
    """Fetch prices for multiple cryptocurrencies."""
    results = {}
    for code in crypto_codes:
        results[code] = get_crypto_price(code)
    return results

def get_crypto_buy_price(crypto_code):
    """Fetch the buy price for a cryptocurrency."""
    url = f"{API_BASE_URL}/{crypto_code}/buy"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('data', {})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching buy price for {crypto_code}: {e}")
        return None
'''

# Create the test_crypto.py file
test_crypto_code = '''#!/usr/bin/env python3

import pytest
import crypto_app

def test_get_single_crypto_price():
    """Test fetching a single cryptocurrency price."""
    result = crypto_app.get_crypto_price('BTC-USD')
    assert result is not None, "Result should not be None"
    assert 'amount' in result, "Result should contain 'amount' field"
    assert 'currency' in result, "Result should contain 'currency' field"
    assert result['currency'] == 'USD', "Currency should be USD"
    # Verify amount is a valid number
    amount = float(result['amount'])
    assert amount > 0, "Price should be positive"

def test_get_ethereum_price():
    """Test fetching Ethereum price."""
    result = crypto_app.get_crypto_price('ETH-USD')
    assert result is not None
    assert 'amount' in result
    assert 'currency' in result
    amount = float(result['amount'])
    assert amount > 0

def test_get_multiple_prices():
    """Test fetching multiple cryptocurrency prices."""
    crypto_codes = ['BTC-USD', 'ETH-USD']
    results = crypto_app.get_multiple_prices(crypto_codes)
    
    assert results is not None
    assert len(results) == 2
    
    for code in crypto_codes:
        assert code in results
        assert results[code] is not None
        assert 'amount' in results[code]
        assert 'currency' in results[code]
        amount = float(results[code]['amount'])
        assert amount > 0

def test_get_crypto_buy_price():
    """Test fetching buy price for a cryptocurrency."""
    result = crypto_app.get_crypto_buy_price('BTC-USD')
    assert result is not None
    assert 'amount' in result
    assert 'currency' in result
    amount = float(result['amount'])
    assert amount > 0

def test_invalid_crypto_code():
    """Test error handling for invalid cryptocurrency code."""
    result = crypto_app.get_crypto_price('INVALID-FAKE-CODE')
    # The API might return None or an error structure
    # This test verifies error handling works without crashing
    assert result is None or 'amount' not in result
'''

# Create the vcr_setup.py file
vcr_setup_code = '''#!/usr/bin/env python3

import vcr
import os

# Define the cassette directory
CASSETTE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cassettes')

# Ensure the cassette directory exists
os.makedirs(CASSETTE_DIR, exist_ok=True)

# Configure VCR with default settings
my_vcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode='once',  # Record on first run, replay on subsequent runs
    match_on=['uri', 'method'],
    filter_headers=['authorization'],
    decode_compressed_response=True,
)

def get_cassette_path():
    """Return the absolute path to the cassette directory."""
    return os.path.abspath(CASSETTE_DIR)

# Export the configured VCR instance
__all__ = ['my_vcr', 'CASSETTE_DIR', 'get_cassette_path']
'''

# Create conftest.py to integrate VCR with pytest
conftest_code = '''#!/usr/bin/env python3

import pytest
import vcr_setup

@pytest.fixture(scope='function', autouse=True)
def vcr_cassette(request):
    """Automatically wrap each test with VCR recording/playback."""
    # Generate cassette name from test name
    cassette_name = f"{request.node.name}.yaml"
    
    # Use VCR context manager for the test
    with vcr_setup.my_vcr.use_cassette(cassette_name):
        yield
'''

def main():
    workspace_dir = '/workspace'
    
    # Ensure workspace directory exists
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Write the files
    with open(os.path.join(workspace_dir, 'crypto_app.py'), 'w') as f:
        f.write(crypto_app_code)
    
    with open(os.path.join(workspace_dir, 'test_crypto.py'), 'w') as f:
        f.write(test_crypto_code)
    
    with open(os.path.join(workspace_dir, 'vcr_setup.py'), 'w') as f:
        f.write(vcr_setup_code)
    
    with open(os.path.join(workspace_dir, 'conftest.py'), 'w') as f:
        f.write(conftest_code)
    
    # Change to workspace directory
    os.chdir(workspace_dir)
    
    # Install required packages
    print("Installing required packages...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytest', 'requests', 'vcrpy'], check=True)
    
    # Remove any existing cassettes
    cassette_dir = os.path.join(workspace_dir, 'cassettes')
    if os.path.exists(cassette_dir):
        shutil.rmtree(cassette_dir)
    
    # First run - record HTTP interactions
    print("\n=== First Run (Recording) ===")
    first_run_result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'test_crypto.py', '-v'],
        cwd=workspace_dir,
        capture_output=True,
        text=True
    )
    print(first_run_result.stdout)
    print(first_run_result.stderr)
    first_run_passed = first_run_result.returncode == 0
    
    print(f"\nFirst run passed: {first_run_passed}")
    print(f"Cassettes created: {os.path.exists(cassette_dir) and len(os.listdir(cassette_dir)) > 0}")
    
    # Second run - replay recorded interactions
    print("\n=== Second Run (Replay) ===")
    second_run_result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'test_crypto.py', '-v'],
        cwd=workspace_dir,
        capture_output=True,
        text=True
    )
    print(second_run_result.stdout)
    print(second_run_result.stderr)
    second_run_passed = second_run_result.returncode == 0
    
    print(f"\nSecond run passed: {second_run_passed}")
    
    # Write results file
    cassette_path = os.path.abspath(cassette_dir)
    results_content = f"""cassette_path={cassette_path}
first_run_passed={str(first_run_passed).lower()}
second_run_passed={str(second_run_passed).lower()}
"""
    
    with open(os.path.join(workspace_dir, 'results.txt'), 'w') as f:
        f.write(results_content)
    
    print(f"\n=== Results ===")
    print(f"Cassette path: {cassette_path}")
    print(f"First run passed: {first_run_passed}")
    print(f"Second run passed: {second_run_passed}")
    
    # Verify cassettes were created
    if os.path.exists(cassette_dir):
        cassettes = os.listdir(cassette_dir)
        print(f"Cassettes created: {len(cassettes)}")
        for cassette in cassettes:
            print(f"  - {cassette}")
    
    return 0 if (first_run_passed and second_run_passed) else 1

if __name__ == '__main__':
    sys.exit(main())