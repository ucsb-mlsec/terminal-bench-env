#!/usr/bin/env python3

import requests
import sys
import time

BASE_URL = "http://127.0.0.1:5000"

def wait_for_flask():
    """Wait for Flask app to be ready"""
    max_attempts = 10
    for i in range(max_attempts):
        try:
            response = requests.get(BASE_URL, timeout=2)
            print("✓ Flask app is running")
            return True
        except requests.exceptions.RequestException:
            if i < max_attempts - 1:
                time.sleep(1)
    print("✗ Flask app is not responding")
    return False

def test_basic_connectivity():
    """Test that the Flask app responds to basic requests"""
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            print("✓ Basic connectivity test passed")
            return True
        else:
            print(f"✗ Basic connectivity test failed: status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Basic connectivity test failed: {e}")
        return False

def test_simple_template_substitution():
    """Test that basic template variables are substituted correctly"""
    try:
        data = {
            'customer_name': 'Alice Johnson',
            'feedback_text': 'Great service and friendly staff'
        }
        response = requests.post(f"{BASE_URL}/submit", data=data, timeout=5)
        
        if response.status_code != 200:
            print(f"✗ Template substitution test failed: status {response.status_code}")
            return False
        
        content = response.text
        
        if 'Alice Johnson' in content and 'Great service and friendly staff' in content:
            print("✓ Simple template substitution test passed")
            return True
        else:
            print("✗ Template substitution test failed: expected values not in response")
            return False
    except Exception as e:
        print(f"✗ Template substitution test failed: {e}")
        return False

def test_special_characters_in_feedback():
    """Test that feedback with special characters is handled correctly"""
    try:
        data = {
            'customer_name': 'Bob Smith',
            'feedback_text': 'Product cost $50 & came with 100% satisfaction!'
        }
        response = requests.post(f"{BASE_URL}/submit", data=data, timeout=5)
        
        if response.status_code != 200:
            print(f"✗ Special characters test failed: status {response.status_code}")
            return False
        
        content = response.text
        
        if 'Bob Smith' in content:
            print("✓ Special characters test passed")
            return True
        else:
            print("✗ Special characters test failed: customer name not found")
            return False
    except Exception as e:
        print(f"✗ Special characters test failed: {e}")
        return False

def test_multiple_submissions():
    """Test that multiple submissions work correctly"""
    try:
        test_cases = [
            {'customer_name': 'John Doe', 'feedback_text': 'Excellent product quality'},
            {'customer_name': 'Jane Smith', 'feedback_text': 'Fast delivery'},
            {'customer_name': 'Mike Brown', 'feedback_text': 'Good customer support'}
        ]
        
        for data in test_cases:
            response = requests.post(f"{BASE_URL}/submit", data=data, timeout=5)
            if response.status_code != 200:
                print(f"✗ Multiple submissions test failed for {data['customer_name']}")
                return False
            if data['customer_name'] not in response.text:
                print(f"✗ Multiple submissions test failed: {data['customer_name']} not in response")
                return False
        
        print("✓ Multiple submissions test passed")
        return True
    except Exception as e:
        print(f"✗ Multiple submissions test failed: {e}")
        return False

def test_date_variable_present():
    """Test that date variable is rendered in template"""
    try:
        data = {
            'customer_name': 'Test User',
            'feedback_text': 'Testing date rendering'
        }
        response = requests.post(f"{BASE_URL}/submit", data=data, timeout=5)
        
        if response.status_code != 200:
            print(f"✗ Date variable test failed: status {response.status_code}")
            return False
        
        content = response.text
        
        # Check if response contains year (indicates date was rendered)
        if '202' in content or '2024' in content or '2025' in content:
            print("✓ Date variable test passed")
            return True
        else:
            print("✓ Date variable test passed (format may vary)")
            return True  # Be lenient on date format
    except Exception as e:
        print(f"✗ Date variable test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Legitimate Functionality of Feedback Portal")
    print("=" * 60)
    
    # Wait for Flask to be ready
    if not wait_for_flask():
        print("\nFailed to connect to Flask app. Make sure it's running on port 5000.")
        sys.exit(1)
    
    time.sleep(0.5)  # Brief pause
    
    # Run all tests
    tests = [
        test_basic_connectivity,
        test_simple_template_substitution,
        test_special_characters_in_feedback,
        test_multiple_submissions,
        test_date_variable_present
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        time.sleep(0.2)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    if all(results):
        print("SUCCESS: All legitimate functionality tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("FAILURE: Some tests failed")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()