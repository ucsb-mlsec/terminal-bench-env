#!/usr/bin/env python3

import os
import sys
import json

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def run_tests():
    """Run all security tests and return True if all pass"""
    client = app.test_client()
    all_passed = True
    test_count = 0
    passed_count = 0
    
    # Test 1: Attempt to download file with ../ traversal
    test_count += 1
    print("Test 1: Download with ../ traversal...", end=" ")
    response = client.get('/download?file=../../../etc/passwd')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Test 2: Attempt to view file with absolute path
    test_count += 1
    print("Test 2: View with absolute path...", end=" ")
    response = client.get('/view?file=/etc/passwd')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Test 3: Attempt URL encoded traversal
    test_count += 1
    print("Test 3: URL encoded traversal...", end=" ")
    response = client.get('/download?file=..%2f..%2f..%2fetc%2fpasswd')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Test 4: Attempt double encoded traversal
    test_count += 1
    print("Test 4: Double encoded traversal...", end=" ")
    response = client.get('/view?file=....//....//etc/passwd')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Test 5: Attempt backslash traversal (Windows style)
    test_count += 1
    print("Test 5: Backslash traversal...", end=" ")
    response = client.get('/download?file=..\\..\\..\\etc\\passwd')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Test 6: Attempt to access system files
    test_count += 1
    print("Test 6: Access system file...", end=" ")
    response = client.get('/view?file=../../../../../../etc/hosts')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Test 7: Null byte injection
    test_count += 1
    print("Test 7: Null byte injection...", end=" ")
    response = client.get('/download?file=../../../etc/passwd%00.txt')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Test 8: Traversal with mixed separators
    test_count += 1
    print("Test 8: Mixed separators...", end=" ")
    response = client.get('/view?file=..\\/../../../etc/passwd')
    if response.status_code in [400, 403, 404]:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Positive Test 1: Create a legitimate test file
    test_count += 1
    print("Test 9: Create legitimate test file...", end=" ")
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'documents')
    os.makedirs(docs_dir, exist_ok=True)
    test_file = os.path.join(docs_dir, 'test_document.txt')
    with open(test_file, 'w') as f:
        f.write('This is a test document')
    print("PASS")
    passed_count += 1
    
    # Positive Test 2: Verify legitimate file access works
    test_count += 1
    print("Test 10: Access legitimate file...", end=" ")
    response = client.get('/view?file=test_document.txt')
    if response.status_code == 200:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    # Positive Test 3: Download legitimate file
    test_count += 1
    print("Test 11: Download legitimate file...", end=" ")
    response = client.get('/download?file=test_document.txt')
    if response.status_code == 200:
        print("PASS")
        passed_count += 1
    else:
        print(f"FAIL (status: {response.status_code})")
        all_passed = False
    
    print(f"\nResults: {passed_count}/{test_count} tests passed")
    
    if all_passed:
        print("\n=== PASS ===")
        return True
    else:
        print("\n=== FAIL ===")
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)