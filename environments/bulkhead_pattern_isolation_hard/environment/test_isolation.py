#!/usr/bin/env python3

import requests
import time
import json
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor

def make_request(endpoint):
    """Make a request to the specified endpoint and return timing/status info."""
    start_time = time.time()
    try:
        response = requests.get(f"http://localhost:8000/{endpoint}", timeout=15)
        end_time = time.time()
        duration = end_time - start_time
        return {
            "endpoint": endpoint,
            "duration": duration,
            "success": True,
            "status_code": response.status_code
        }
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        return {
            "endpoint": endpoint,
            "duration": duration,
            "success": False,
            "status_code": None,
            "error": str(e)
        }

def main():
    print("Starting FastAPI server...")
    server_process = subprocess.Popen(
        [sys.executable, "/app/main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Waiting for server to start...")
    time.sleep(2)
    
    print("\nMaking concurrent requests to all endpoints...")
    print("=" * 60)
    
    # Make concurrent requests to all three endpoints
    with ThreadPoolExecutor(max_workers=3) as executor:
        endpoints = ["auth", "data", "reports"]
        futures = [executor.submit(make_request, endpoint) for endpoint in endpoints]
        results = [future.result() for future in futures]
    
    print("\nResults:")
    print("-" * 60)
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['endpoint']:10s} - Duration: {result['duration']:.2f}s - "
              f"Status: {result.get('status_code', 'N/A')}")
    
    # Verify results
    auth_result = next(r for r in results if r["endpoint"] == "auth")
    data_result = next(r for r in results if r["endpoint"] == "data")
    
    print("\n" + "=" * 60)
    print("Verification:")
    print("-" * 60)
    
    auth_fast = auth_result["duration"] < 1.0
    data_fast = data_result["duration"] < 1.0
    auth_success = auth_result["success"] and auth_result["status_code"] == 200
    data_success = data_result["success"] and data_result["status_code"] == 200
    
    print(f"Auth endpoint < 1s: {auth_fast} ({auth_result['duration']:.2f}s)")
    print(f"Data endpoint < 1s: {data_fast} ({data_result['duration']:.2f}s)")
    print(f"Auth endpoint status 200: {auth_success}")
    print(f"Data endpoint status 200: {data_success}")
    
    isolation_implemented = auth_fast and data_fast
    test_passed = auth_fast and data_fast and auth_success and data_success
    
    # Write results to file
    result_data = {
        "isolation_implemented": isolation_implemented,
        "test_passed": test_passed
    }
    
    with open("/app/result.json", "w") as f:
        json.dump(result_data, f, indent=2)
    
    print("\n" + "=" * 60)
    if test_passed:
        print("✓ TEST PASSED: Isolation is working correctly!")
        print("  Auth and data endpoints remained responsive during reports timeout.")
    else:
        print("✗ TEST FAILED: Isolation not working properly.")
        if not isolation_implemented:
            print("  Auth and/or data endpoints were blocked by reports endpoint.")
    
    print(f"\nResults written to /app/result.json")
    print("=" * 60)
    
    # Clean up
    print("\nShutting down server...")
    server_process.terminate()
    server_process.wait(timeout=5)
    
    # Exit with appropriate code
    sys.exit(0 if test_passed else 1)

if __name__ == "__main__":
    main()