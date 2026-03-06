#!/usr/bin/env python3

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Pre-populate product data for IDs 1001-1200
PRODUCTS = {}
prices = [19.99, 29.99, 39.99, 49.99, 59.99, 69.99, 79.99, 89.99, 99.99, 119.99, 139.99, 159.99, 179.99, 199.99]
for i in range(1001, 1201):
    PRODUCTS[str(i)] = {
        "id": str(i),
        "name": f"Product {i}",
        "price": prices[(i - 1001) % len(prices)]
    }

@app.route('/graphql', methods=['POST'])
def graphql():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Basic GraphQL query parsing for products(ids: [...])
        if 'products' in query and 'ids:' in query:
            # Extract IDs from query
            start = query.find('ids:')
            if start == -1:
                return jsonify({"errors": [{"message": "Invalid query format"}]}), 400
            
            # Find the array of IDs
            bracket_start = query.find('[', start)
            bracket_end = query.find(']', bracket_start)
            
            if bracket_start == -1 or bracket_end == -1:
                return jsonify({"errors": [{"message": "Invalid query format"}]}), 400
            
            ids_str = query[bracket_start+1:bracket_end]
            # Parse IDs - remove quotes and whitespace
            ids = [id.strip().strip('"').strip("'") for id in ids_str.split(',') if id.strip()]
            
            # Enforce rate limit: max 25 products per query
            if len(ids) > 25:
                return jsonify({
                    "errors": [{
                        "message": f"Rate limit exceeded: requested {len(ids)} products, maximum is 25 per query"
                    }]
                }), 400
            
            # Fetch products
            products = []
            for product_id in ids:
                if product_id in PRODUCTS:
                    products.append(PRODUCTS[product_id])
            
            return jsonify({
                "data": {
                    "products": products
                }
            })
        
        return jsonify({"errors": [{"message": "Unsupported query"}]}), 400
        
    except Exception as e:
        return jsonify({"errors": [{"message": str(e)}]}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)