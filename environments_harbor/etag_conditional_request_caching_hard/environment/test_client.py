#!/usr/bin/env python3

import hashlib
import json
import sqlite3
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)
DATABASE = '/home/api/product_service/database.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def generate_etag(data):
    """Generate ETag from data content"""
    content = json.dumps(data, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

@app.route('/products', methods=['GET'])
def get_products():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products')
    rows = cursor.fetchall()
    conn.close()
    
    products = [dict(row) for row in rows]
    etag = generate_etag(products)
    
    # Check If-None-Match header
    if_none_match = request.headers.get('If-None-Match')
    if if_none_match and if_none_match == etag:
        response = make_response('', 304)
        response.headers['ETag'] = etag
        return response
    
    response = make_response(jsonify(products), 200)
    response.headers['ETag'] = etag
    return response

@app.route('/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return jsonify({'error': 'Product not found'}), 404
    
    product = dict(row)
    etag = generate_etag(product)
    
    # Check If-None-Match header
    if_none_match = request.headers.get('If-None-Match')
    if if_none_match and if_none_match == etag:
        response = make_response('', 304)
        response.headers['ETag'] = etag
        return response
    
    response = make_response(jsonify(product), 200)
    response.headers['ETag'] = etag
    return response

@app.route('/products/<int:product_id>', methods=['PUT'])
def update_product(product_id):
    conn = get_db()
    cursor = conn.cursor()
    
    # Get current product state
    cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
    row = cursor.fetchone()
    
    if row is None:
        conn.close()
        return jsonify({'error': 'Product not found'}), 404
    
    current_product = dict(row)
    current_etag = generate_etag(current_product)
    
    # Check If-Match header
    if_match = request.headers.get('If-Match')
    if if_match and if_match != current_etag:
        conn.close()
        return jsonify({'error': 'Precondition Failed'}), 412
    
    # Update the product
    data = request.get_json()
    update_fields = []
    values = []
    
    if 'name' in data:
        update_fields.append('name = ?')
        values.append(data['name'])
    if 'price' in data:
        update_fields.append('price = ?')
        values.append(data['price'])
    if 'description' in data:
        update_fields.append('description = ?')
        values.append(data['description'])
    
    if update_fields:
        values.append(product_id)
        query = f"UPDATE products SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, values)
        conn.commit()
    
    # Get updated product
    cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
    row = cursor.fetchone()
    updated_product = dict(row)
    conn.close()
    
    new_etag = generate_etag(updated_product)
    response = make_response(jsonify(updated_product), 200)
    response.headers['ETag'] = new_etag
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

```python
#!/usr/bin/env python3

import requests
import time
import sys

BASE_URL = 'http://localhost:5000'

def run_tests():
    results = []
    
    # Test 1: GET request returns an ETag header
    print("Test 1: GET /products returns ETag header")
    try:
        response = requests.get(f'{BASE_URL}/products')
        if response.status_code == 200 and 'ETag' in response.headers:
            print("  PASS: Got 200 with ETag header")
            results.append(True)
            initial_etag = response.headers['ETag']
        else:
            print("  FAIL: Missing ETag header or wrong status")
            results.append(False)
            initial_etag = None
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)
        initial_etag = None
    
    # Test 2: Subsequent GET with If-None-Match matching returns 304
    print("\nTest 2: GET with matching If-None-Match returns 304")
    try:
        if initial_etag:
            response = requests.get(f'{BASE_URL}/products', 
                                   headers={'If-None-Match': initial_etag})
            if response.status_code == 304:
                print("  PASS: Got 304 Not Modified")
                results.append(True)
            else:
                print(f"  FAIL: Got {response.status_code} instead of 304")
                results.append(False)
        else:
            print("  SKIP: No ETag from previous test")
            results.append(False)
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)
    
    # Test 3: GET with If-None-Match that doesn't match returns 200
    print("\nTest 3: GET with non-matching If-None-Match returns 200")
    try:
        response = requests.get(f'{BASE_URL}/products', 
                               headers={'If-None-Match': 'fake-etag-12345'})
        if response.status_code == 200 and 'ETag' in response.headers:
            print("  PASS: Got 200 with ETag")
            results.append(True)
        else:
            print(f"  FAIL: Got {response.status_code} or missing ETag")
            results.append(False)
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)
    
    # Test 4: PUT with If-Match matching current ETag succeeds
    print("\nTest 4: PUT with matching If-Match succeeds")
    try:
        # Get current product state
        response = requests.get(f'{BASE_URL}/products/1')
        if response.status_code == 200 and 'ETag' in response.headers:
            product_etag = response.headers['ETag']
            original_data = response.json()
            
            # Update with matching ETag
            update_data = {'name': 'Updated Product'}
            response = requests.put(f'{BASE_URL}/products/1', 
                                   json=update_data,
                                   headers={'If-Match': product_etag})
            if response.status_code == 200:
                print("  PASS: Update succeeded with matching ETag")
                results.append(True)
            else:
                print(f"  FAIL: Got {response.status_code} instead of 200")
                results.append(False)
        else:
            print("  FAIL: Could not get initial product state")
            results.append(False)
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)
    
    # Test 5: PUT with If-Match not matching returns 412
    print("\nTest 5: PUT with non-matching If-Match returns 412")
    try:
        update_data = {'name': 'Another Update'}
        response = requests.put(f'{BASE_URL}/products/1', 
                               json=update_data,
                               headers={'If-Match': 'wrong-etag-xyz'})
        if response.status_code == 412:
            print("  PASS: Got 412 Precondition Failed")
            results.append(True)
        else:
            print(f"  FAIL: Got {response.status_code} instead of 412")
            results.append(False)
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)
    
    # Test 6: GET without If-None-Match returns 200 (backward compatibility)
    print("\nTest 6: GET without If-None-Match returns 200 (backward compatibility)")
    try:
        response = requests.get(f'{BASE_URL}/products/1')
        if response.status_code == 200 and 'ETag' in response.headers:
            print("  PASS: Got 200 with ETag (backward compatible)")
            results.append(True)
        else:
            print(f"  FAIL: Got {response.status_code} or missing ETag")
            results.append(False)
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)
    
    return results

def main():
    print("Starting ETag API Tests\n")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    results = run_tests()
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    overall = "PASS" if failed == 0 else "FAIL"
    
    print("\n" + "=" * 60)
    print(f"\nTest Summary:")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Overall: {overall}")
    
    # Write results to file
    with open('/home/api/test_results.txt', 'w') as f:
        f.write(f"{total}\n")
        f.write(f"{passed}\n")
        f.write(f"{failed}\n")
        f.write(f"{overall}\n")
    
    print(f"\nResults written to /home/api/test_results.txt")
    
    sys.exit(0 if overall == "PASS" else 1)

if __name__ == '__main__':
    main()