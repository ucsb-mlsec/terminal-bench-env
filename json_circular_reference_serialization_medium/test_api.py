#!/usr/bin/env python3

import requests
import json
import time
import sys

def test_api():
    """Test the Flask API endpoint for team data."""
    
    api_url = 'http://localhost:5000/api/team/1'
    max_retries = 5
    retry_delay = 2
    
    print("=" * 60)
    print("Testing Flask API Service")
    print("=" * 60)
    print(f"\nAPI Endpoint: {api_url}")
    print(f"Max retries: {max_retries}")
    print(f"Retry delay: {retry_delay} seconds\n")
    
    # Wait and retry mechanism to allow service to start
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}...")
        
        try:
            # Make GET request to the endpoint
            response = requests.get(api_url, timeout=5)
            
            # Check status code
            print(f"Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"ERROR: Expected status code 200, got {response.status_code}")
                print(f"Response: {response.text}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                    continue
                return False
            
            # Verify response is valid JSON
            try:
                data = response.json()
                print("✓ Response is valid JSON")
            except json.JSONDecodeError as e:
                print(f"ERROR: Response is not valid JSON: {e}")
                print(f"Response text: {response.text[:200]}")
                return False
            
            # Print the response data
            print("\n" + "-" * 60)
            print("Response Data:")
            print("-" * 60)
            print(json.dumps(data, indent=2))
            print("-" * 60 + "\n")
            
            # Check for expected fields
            print("Validating response structure...")
            
            if 'id' not in data:
                print("ERROR: Missing 'id' field in response")
                return False
            print(f"✓ Field 'id' found: {data['id']}")
            
            if 'name' not in data:
                print("ERROR: Missing 'name' field in response")
                return False
            print(f"✓ Field 'name' found: {data['name']}")
            
            if 'members' not in data:
                print("ERROR: Missing 'members' field in response")
                return False
            print(f"✓ Field 'members' found: {len(data['members'])} members")
            
            # Validate members structure
            if not isinstance(data['members'], list):
                print("ERROR: 'members' field is not a list")
                return False
            print("✓ Members is a list")
            
            if len(data['members']) > 0:
                member = data['members'][0]
                required_member_fields = ['id', 'name', 'role']
                for field in required_member_fields:
                    if field not in member:
                        print(f"ERROR: Missing '{field}' field in member data")
                        return False
                print(f"✓ Member structure valid (checked {len(required_member_fields)} fields)")
            
            # Check for circular references by ensuring JSON can be serialized again
            try:
                json.dumps(data)
                print("✓ No circular references detected (JSON re-serializable)")
            except (TypeError, ValueError) as e:
                print(f"ERROR: Circular reference or serialization issue: {e}")
                return False
            
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED!")
            print("=" * 60)
            return True
            
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Could not connect to the service at {api_url}")
            if attempt < max_retries:
                print(f"Service may not be ready yet. Retrying in {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                print("ERROR: Service failed to start after all retries")
                return False
                
        except requests.exceptions.Timeout:
            print("ERROR: Request timed out")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"ERROR: HTTP request failed: {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                return False
                
        except Exception as e:
            print(f"ERROR: Unexpected error: {e}")
            return False
    
    return False

if __name__ == "__main__":
    print("\nStarting API test...\n")
    
    success = test_api()
    
    if success:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)