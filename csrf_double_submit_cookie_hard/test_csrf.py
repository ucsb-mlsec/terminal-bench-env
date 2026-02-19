#!/usr/bin/env python3

import requests
import sys
import re
from http.cookies import SimpleCookie

BASE_URL = 'http://localhost:5000'

def print_test(message):
    print(f"[TEST] {message}")

def print_pass(message):
    print(f"[PASS] {message}")

def print_fail(message):
    print(f"[FAIL] {message}")

class CSRFTester:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.session = requests.Session()

    def test_token_randomness(self):
        """Test that tokens are random and different on each request"""
        print_test("Testing token randomness...")
        try:
            tokens = []
            for i in range(5):
                response = self.session.get(f"{BASE_URL}/")
                if response.status_code != 200:
                    print_fail(f"Could not fetch form (status {response.status_code})")
                    self.tests_failed += 1
                    return False
                
                # Extract token from cookie
                if 'csrf_token' in response.cookies:
                    tokens.append(response.cookies['csrf_token'])
                else:
                    print_fail("No csrf_token cookie found in response")
                    self.tests_failed += 1
                    return False
            
            # Check all tokens are different
            if len(set(tokens)) == len(tokens):
                print_pass("Tokens are random and unique")
                self.tests_passed += 1
                return True
            else:
                print_fail("Tokens are not sufficiently random")
                self.tests_failed += 1
                return False
        except Exception as e:
            print_fail(f"Token randomness test failed: {e}")
            self.tests_failed += 1
            return False

    def test_cookie_security_attributes(self):
        """Test that cookies have proper security attributes"""
        print_test("Testing cookie security attributes...")
        try:
            response = self.session.get(f"{BASE_URL}/")
            
            # Check Set-Cookie header
            set_cookie_header = response.headers.get('Set-Cookie', '')
            
            has_httponly = 'HttpOnly' in set_cookie_header
            has_samesite = 'SameSite' in set_cookie_header
            
            if has_httponly and has_samesite:
                print_pass("Cookies have HttpOnly and SameSite attributes")
                self.tests_passed += 1
                return True, "SECURE"
            else:
                missing = []
                if not has_httponly:
                    missing.append("HttpOnly")
                if not has_samesite:
                    missing.append("SameSite")
                print_fail(f"Cookies missing security attributes: {', '.join(missing)}")
                self.tests_failed += 1
                return False, "INSECURE"
        except Exception as e:
            print_fail(f"Cookie security test failed: {e}")
            self.tests_failed += 1
            return False, "INSECURE"

    def test_valid_token_submission(self):
        """Test that valid token submission succeeds"""
        print_test("Testing valid token submission to /submit...")
        try:
            # Get form and token
            response = self.session.get(f"{BASE_URL}/")
            token_from_cookie = response.cookies.get('csrf_token')
            
            # Extract token from HTML
            match = re.search(r'name=["\']csrf_token["\'] value=["\']([^"\']+)["\']', response.text)
            if not match:
                match = re.search(r'value=["\']([^"\']+)["\'] name=["\']csrf_token["\']', response.text)
            
            if not match:
                print_fail("Could not find csrf_token in form HTML")
                self.tests_failed += 1
                return False
            
            token_from_form = match.group(1)
            
            # Submit with valid token
            post_response = self.session.post(
                f"{BASE_URL}/submit",
                data={'csrf_token': token_from_cookie, 'data': 'test'},
                cookies={'csrf_token': token_from_cookie}
            )
            
            if post_response.status_code == 200:
                print_pass("Valid token submission to /submit succeeded")
                self.tests_passed += 1
                return True
            else:
                print_fail(f"Valid token submission failed with status {post_response.status_code}")
                self.tests_failed += 1
                return False
        except Exception as e:
            print_fail(f"Valid token test failed: {e}")
            self.tests_failed += 1
            return False

    def test_missing_token(self):
        """Test that requests without tokens are rejected"""
        print_test("Testing POST without token to /submit...")
        try:
            response = requests.post(f"{BASE_URL}/submit", data={'data': 'test'})
            
            if response.status_code in [400, 403]:
                print_pass("Request without token correctly rejected")
                self.tests_passed += 1
                return True
            else:
                print_fail(f"Request without token not rejected (status {response.status_code})")
                self.tests_failed += 1
                return False
        except Exception as e:
            print_fail(f"Missing token test failed: {e}")
            self.tests_failed += 1
            return False

    def test_mismatched_token(self):
        """Test that requests with mismatched tokens are rejected"""
        print_test("Testing POST with mismatched token to /submit...")
        try:
            # Get valid token
            response = self.session.get(f"{BASE_URL}/")
            valid_token = response.cookies.get('csrf_token')
            
            # Submit with different token in form data
            post_response = requests.post(
                f"{BASE_URL}/submit",
                data={'csrf_token': 'invalid_token_12345', 'data': 'test'},
                cookies={'csrf_token': valid_token}
            )
            
            if post_response.status_code in [400, 403]:
                print_pass("Request with mismatched token correctly rejected")
                self.tests_passed += 1
                return True
            else:
                print_fail(f"Request with mismatched token not rejected (status {post_response.status_code})")
                self.tests_failed += 1
                return False
        except Exception as e:
            print_fail(f"Mismatched token test failed: {e}")
            self.tests_failed += 1
            return False

    def test_delete_endpoint_protection(self):
        """Test that /delete endpoint is also protected"""
        print_test("Testing CSRF protection on /delete endpoint...")
        try:
            # Test without token
            response = requests.post(f"{BASE_URL}/delete", data={'id': '1'})
            if response.status_code not in [400, 403]:
                print_fail(f"/delete endpoint not protected (status {response.status_code})")
                self.tests_failed += 1
                return False
            
            # Test with valid token
            get_response = self.session.get(f"{BASE_URL}/")
            token = get_response.cookies.get('csrf_token')
            
            post_response = self.session.post(
                f"{BASE_URL}/delete",
                data={'csrf_token': token, 'id': '1'},
                cookies={'csrf_token': token}
            )
            
            if post_response.status_code == 200:
                print_pass("/delete endpoint properly protected and accepts valid tokens")
                self.tests_passed += 1
                return True
            else:
                print_fail(f"/delete endpoint rejected valid token (status {post_response.status_code})")
                self.tests_failed += 1
                return False
        except Exception as e:
            print_fail(f"/delete endpoint test failed: {e}")
            self.tests_failed += 1
            return False

    def run_all_tests(self):
        """Run all tests and return results"""
        print("\n" + "="*60)
        print("Starting CSRF Protection Tests")
        print("="*60 + "\n")
        
        # Test 1: Token randomness
        self.test_token_randomness()
        print()
        
        # Test 2: Cookie security
        cookie_test_passed, security_status = self.test_cookie_security_attributes()
        print()
        
        # Test 3: Valid token
        self.test_valid_token_submission()
        print()
        
        # Test 4: Missing token
        self.test_missing_token()
        print()
        
        # Test 5: Mismatched token
        self.test_mismatched_token()
        print()
        
        # Test 6: Delete endpoint
        self.test_delete_endpoint_protection()
        print()
        
        print("="*60)
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print("="*60)
        
        all_passed = self.tests_failed == 0
        return all_passed, security_status

def main():
    try:
        # Test if server is running
        try:
            requests.get(f"{BASE_URL}/", timeout=2)
        except requests.exceptions.ConnectionError:
            print_fail("Could not connect to Flask app at http://localhost:5000")
            print("Make sure the Flask application is running")
            sys.exit(1)
        
        tester = CSRFTester()
        all_passed, security_status = tester.run_all_tests()
        
        if all_passed:
            print("\n✓ All tests PASSED")
            sys.exit(0)
        else:
            print("\n✗ Some tests FAILED")
            sys.exit(1)
            
    except Exception as e:
        print_fail(f"Test execution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()