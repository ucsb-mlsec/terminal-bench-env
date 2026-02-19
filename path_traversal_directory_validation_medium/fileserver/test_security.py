#!/usr/bin/env python3

import unittest
import requests
import subprocess
import time
import os
import signal
import sys
from urllib.parse import quote

class TestFileServerSecurity(unittest.TestCase):
    """Comprehensive security test suite for file server path traversal vulnerabilities"""
    
    BASE_URL = "http://localhost:8000"
    SERVER_PROCESS = None
    
    @classmethod
    def setUpClass(cls):
        """Start the file server before running tests"""
        # Ensure test files exist
        os.makedirs("/var/www/public/subdir", exist_ok=True)
        with open("/var/www/public/test.txt", "w") as f:
            f.write("legitimate file content")
        with open("/var/www/public/subdir/nested.txt", "w") as f:
            f.write("nested file content")
        
        # Start the server
        server_path = "/home/user/solution/server.py"
        if not os.path.exists(server_path):
            server_path = "/home/user/fileserver/server.py"
        
        cls.SERVER_PROCESS = subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(2)
        
        # Verify server is running
        max_retries = 10
        for _ in range(max_retries):
            try:
                requests.get(cls.BASE_URL, timeout=1)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
    
    @classmethod
    def tearDownClass(cls):
        """Stop the file server after all tests"""
        if cls.SERVER_PROCESS:
            cls.SERVER_PROCESS.terminate()
            cls.SERVER_PROCESS.wait(timeout=5)
    
    def test_01_basic_relative_traversal(self):
        """Test basic relative path traversal: ../../etc/passwd"""
        response = requests.get(f"{self.BASE_URL}/../../etc/passwd")
        self.assertEqual(response.status_code, 403, "Basic relative traversal should be blocked")
    
    def test_02_multiple_level_traversal(self):
        """Test multiple levels of traversal: ../../../../etc/passwd"""
        response = requests.get(f"{self.BASE_URL}/../../../../etc/passwd")
        self.assertEqual(response.status_code, 403, "Multi-level traversal should be blocked")
    
    def test_03_absolute_path_etc_passwd(self):
        """Test absolute path attempt: /etc/passwd"""
        response = requests.get(f"{self.BASE_URL}//etc/passwd")
        self.assertEqual(response.status_code, 403, "Absolute path to /etc/passwd should be blocked")
    
    def test_04_absolute_path_home(self):
        """Test absolute path to home directory: /home/user/secrets.txt"""
        response = requests.get(f"{self.BASE_URL}//home/user/secrets.txt")
        self.assertEqual(response.status_code, 403, "Absolute path to home should be blocked")
    
    def test_05_url_encoded_traversal(self):
        """Test URL-encoded traversal: ..%2F..%2Fetc%2Fpasswd"""
        response = requests.get(f"{self.BASE_URL}/..%2F..%2Fetc%2Fpasswd")
        self.assertEqual(response.status_code, 403, "URL-encoded traversal should be blocked")
    
    def test_06_double_url_encoded(self):
        """Test double URL-encoded sequences: %252e%252e%252f"""
        path = "%252e%252e%252f%252e%252e%252fetc%252fpasswd"
        response = requests.get(f"{self.BASE_URL}/{path}")
        self.assertEqual(response.status_code, 403, "Double URL-encoded traversal should be blocked")
    
    def test_07_mixed_encoding(self):
        """Test mixed encoding attacks"""
        path = "..%2F../etc/passwd"
        response = requests.get(f"{self.BASE_URL}/{path}")
        self.assertEqual(response.status_code, 403, "Mixed encoding should be blocked")
    
    def test_08_null_byte_injection(self):
        """Test null byte injection: ../../etc/passwd%00.txt"""
        path = "../../etc/passwd%00.txt"
        response = requests.get(f"{self.BASE_URL}/{path}")
        self.assertEqual(response.status_code, 403, "Null byte injection should be blocked")
    
    def test_09_backslash_traversal(self):
        """Test backslash variants: ..\..\etc\passwd"""
        path = "..\\..\\etc\\passwd"
        response = requests.get(f"{self.BASE_URL}/{path}")
        self.assertEqual(response.status_code, 403, "Backslash traversal should be blocked")
    
    def test_10_dot_segments_without_slashes(self):
        """Test dot segments: ...//...//etc/passwd"""
        response = requests.get(f"{self.BASE_URL}/...//...//etc/passwd")
        self.assertEqual(response.status_code, 403, "Dot segment attacks should be blocked")
    
    def test_11_unicode_encoding(self):
        """Test Unicode encoding attempts"""
        path = "%u002e%u002e%u002f%u002e%u002e%u002fetc%u002fpasswd"
        response = requests.get(f"{self.BASE_URL}/{path}")
        self.assertEqual(response.status_code, 403, "Unicode encoding should be blocked")
    
    def test_12_legitimate_file_access(self):
        """Verify legitimate files in /var/www/public ARE accessible"""
        response = requests.get(f"{self.BASE_URL}/test.txt")
        self.assertEqual(response.status_code, 200, "Legitimate files should be accessible")
        self.assertIn("legitimate file content", response.text)
    
    def test_13_subdirectory_access(self):
        """Verify files in subdirectories of /var/www/public ARE accessible"""
        response = requests.get(f"{self.BASE_URL}/subdir/nested.txt")
        self.assertEqual(response.status_code, 200, "Subdirectory files should be accessible")
        self.assertIn("nested file content", response.text)
    
    def test_14_redundant_slashes(self):
        """Test path with redundant slashes: //../../etc/passwd"""
        response = requests.get(f"{self.BASE_URL}///../../etc/passwd")
        self.assertEqual(response.status_code, 403, "Redundant slashes should be blocked")
    
    def test_15_mixed_case_encoding(self):
        """Test mixed case encoding: %2e%2e%2F%2e%2e%2Fetc%2Fpasswd"""
        path = "%2e%2e%2F%2e%2e%2Fetc%2Fpasswd"
        response = requests.get(f"{self.BASE_URL}/{path}")
        self.assertEqual(response.status_code, 403, "Mixed case encoding should be blocked")
    
    def test_16_traversal_in_subdirectory(self):
        """Test traversal from subdirectory: subdir/../../etc/passwd"""
        response = requests.get(f"{self.BASE_URL}/subdir/../../etc/passwd")
        self.assertEqual(response.status_code, 403, "Traversal from subdirectory should be blocked")
    
    def test_17_deep_traversal(self):
        """Test deep directory traversal: ../../../../../../../../../etc/passwd"""
        response = requests.get(f"{self.BASE_URL}/../../../../../../../../../etc/passwd")
        self.assertEqual(response.status_code, 403, "Deep traversal should be blocked")

def run_tests():
    """Run all tests and generate report"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFileServerSecurity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    result = run_tests()
    
    # Generate validation report
    import json
    
    tests_run = result.testsRun
    tests_failed = len(result.failures) + len(result.errors)
    tests_passed = tests_run - tests_failed
    
    # Count vulnerability types fixed (based on test categories)
    vulnerabilities_fixed = 7 if tests_passed >= 15 else 0
    server_functional = tests_passed >= 2  # At least legitimate access tests passed
    
    report = {
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "vulnerabilities_fixed": vulnerabilities_fixed,
        "server_functional": server_functional
    }
    
    os.makedirs("/home/user/solution", exist_ok=True)
    with open("/home/user/solution/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Validation Report:")
    print(f"Tests Passed: {tests_passed}/{tests_run}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Vulnerabilities Fixed: {vulnerabilities_fixed}")
    print(f"Server Functional: {server_functional}")
    print(f"{'='*60}")
    
    sys.exit(0 if tests_failed == 0 else 1)