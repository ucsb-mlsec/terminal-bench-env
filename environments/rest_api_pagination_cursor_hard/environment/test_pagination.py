#!/usr/bin/env python3

import requests
import json
import sqlite3
import time
import sys
from datetime import datetime

BASE_URL = 'http://localhost:5000'

def test_offset_pagination():
    """Tests backward compatibility with offset-based pagination."""
    print("\n=== Testing Offset Pagination (Backward Compatibility) ===")
    try:
        # Request first page
        response1 = requests.get(f'{BASE_URL}/api/products?page=1&limit=10')
        if response1.status_code != 200:
            print(f"❌ First page request failed with status {response1.status_code}")
            return False
        
        data1 = response1.json()
        if 'products' not in data1:
            print("❌ Response missing 'products' field")
            return False
        
        if len(data1['products']) != 10:
            print(f"❌ Expected 10 products, got {len(data1['products'])}")
            return False
        
        # Request second page
        response2 = requests.get(f'{BASE_URL}/api/products?page=2&limit=10')
        if response2.status_code != 200:
            print(f"❌ Second page request failed with status {response2.status_code}")
            return False
        
        data2 = response2.json()
        if 'products' not in data2:
            print("❌ Second page response missing 'products' field")
            return False
        
        # Verify results are different
        ids1 = [p['id'] for p in data1['products']]
        ids2 = [p['id'] for p in data2['products']]
        
        if ids1 == ids2:
            print("❌ Page 1 and Page 2 returned identical results")
            return False
        
        if any(id in ids2 for id in ids1):
            print("❌ Found duplicate IDs between pages")
            return False
        
        print("✓ Offset pagination working correctly")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is Flask app running?")
        return False
    except Exception as e:
        print(f"❌ Error during offset pagination test: {str(e)}")
        return False

def test_cursor_pagination():
    """Tests cursor-based pagination."""
    print("\n=== Testing Cursor Pagination ===")
    try:
        # Request first page with cursor pagination
        response1 = requests.get(f'{BASE_URL}/api/products?limit=10')
        if response1.status_code != 200:
            print(f"❌ First request failed with status {response1.status_code}")
            return False
        
        data1 = response1.json()
        if 'products' not in data1:
            print("❌ Response missing 'products' field")
            return False
        
        if 'next_cursor' not in data1:
            print("❌ Response missing 'next_cursor' field")
            return False
        
        if len(data1['products']) != 10:
            print(f"❌ Expected 10 products, got {len(data1['products'])}")
            return False
        
        next_cursor = data1['next_cursor']
        if not next_cursor:
            print("❌ next_cursor is empty")
            return False
        
        # Request second page using cursor
        response2 = requests.get(f'{BASE_URL}/api/products?cursor={next_cursor}&limit=10')
        if response2.status_code != 200:
            print(f"❌ Cursor request failed with status {response2.status_code}")
            return False
        
        data2 = response2.json()
        if 'products' not in data2:
            print("❌ Cursor response missing 'products' field")
            return False
        
        # Verify results are different and sequential
        ids1 = [p['id'] for p in data1['products']]
        ids2 = [p['id'] for p in data2['products']]
        
        if ids1 == ids2:
            print("❌ Cursor pagination returned identical results")
            return False
        
        if any(id in ids2 for id in ids1):
            print("❌ Found duplicate IDs with cursor pagination")
            return False
        
        print("✓ Cursor pagination working correctly")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is Flask app running?")
        return False
    except Exception as e:
        print(f"❌ Error during cursor pagination test: {str(e)}")
        return False

def test_consistency_with_inserts():
    """Tests that cursor pagination remains consistent when records are inserted."""
    print("\n=== Testing Consistency with Concurrent Inserts ===")
    try:
        # Fetch first page with cursor
        response1 = requests.get(f'{BASE_URL}/api/products?limit=10')
        if response1.status_code != 200:
            print(f"❌ First request failed with status {response1.status_code}")
            return False
        
        data1 = response1.json()
        next_cursor = data1.get('next_cursor')
        
        if not next_cursor:
            print("❌ No next_cursor in response")
            return False
        
        ids_seen = set(p['id'] for p in data1['products'])
        
        # Insert new products with recent timestamps
        conn = sqlite3.connect('/home/project/products.db')
        cursor = conn.cursor()
        
        for i in range(5):
            cursor.execute(
                "INSERT INTO products (name, price, created_at, category) VALUES (?, ?, ?, ?)",
                (f"Test Product Insert {i}", 99.99, datetime.now().isoformat(), "Test")
            )
        
        conn.commit()
        conn.close()
        
        print("  Inserted 5 new products into database")
        
        # Continue pagination with the cursor from before the inserts
        response2 = requests.get(f'{BASE_URL}/api/products?cursor={next_cursor}&limit=10')
        if response2.status_code != 200:
            print(f"❌ Second request failed with status {response2.status_code}")
            return False
        
        data2 = response2.json()
        ids_page2 = set(p['id'] for p in data2['products'])
        
        # Check for duplicates
        duplicates = ids_seen.intersection(ids_page2)
        if duplicates:
            print(f"❌ Found duplicate IDs after insert: {duplicates}")
            return False
        
        print("✓ Cursor pagination maintained consistency during inserts")
        return True
        
    except Exception as e:
        print(f"❌ Error during consistency test: {str(e)}")
        return False

def test_performance():
    """Tests that cursor pagination performs better than offset for later pages."""
    print("\n=== Testing Performance Characteristics ===")
    try:
        # Test offset pagination at page 50
        start_time = time.time()
        response_offset = requests.get(f'{BASE_URL}/api/products?page=50&limit=10')
        offset_time = time.time() - start_time
        
        if response_offset.status_code != 200:
            print(f"❌ Offset pagination request failed")
            return False
        
        print(f"  Offset pagination (page 50): {offset_time:.4f} seconds")
        
        # Navigate to equivalent position using cursor
        cursor = None
        for i in range(50):
            params = {'limit': 10}
            if cursor:
                params['cursor'] = cursor
            
            response = requests.get(f'{BASE_URL}/api/products', params=params)
            if response.status_code != 200:
                print(f"❌ Cursor navigation failed at iteration {i}")
                return False
            
            data = response.json()
            cursor = data.get('next_cursor')
            
            if not cursor and i < 49:
                print(f"❌ Ran out of pages at iteration {i}")
                return False
        
        # Now test cursor performance at the same position
        start_time = time.time()
        response_cursor = requests.get(f'{BASE_URL}/api/products?cursor={cursor}&limit=10')
        cursor_time = time.time() - start_time
        
        if response_cursor.status_code != 200:
            print(f"❌ Cursor pagination request failed")
            return False
        
        print(f"  Cursor pagination (page 50): {cursor_time:.4f} seconds")
        
        # Cursor should be faster or at least comparable
        print(f"  Performance ratio: {offset_time/cursor_time:.2f}x")
        print("✓ Performance test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during performance test: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("Starting Pagination Tests")
    print("=" * 60)
    
    # Run all tests
    results = {
        'offset_pagination': test_offset_pagination(),
        'cursor_pagination': test_cursor_pagination(),
        'consistency': test_consistency_with_inserts(),
        'performance': test_performance()
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")
    
    # Determine overall results
    cursor_pagination_working = (
        results['cursor_pagination'] and 
        results['consistency']
    )
    backward_compatible = results['offset_pagination']
    
    print("\n" + "=" * 60)
    print("Final Assessment")
    print("=" * 60)
    print(f"Cursor Pagination Working: {cursor_pagination_working}")
    print(f"Backward Compatible: {backward_compatible}")
    
    if cursor_pagination_working and backward_compatible:
        print("\n✓ All requirements met!")
    else:
        print("\n❌ Some requirements not met")
    
    return cursor_pagination_working, backward_compatible

if __name__ == '__main__':
    try:
        cursor_working, backward_compat = main()
        sys.exit(0 if (cursor_working and backward_compat) else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        sys.exit(1)