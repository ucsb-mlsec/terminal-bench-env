#!/usr/bin/env python3

import random
import json

class DynamoDBSimulator:
    def __init__(self):
        """Initialize simulator with 5000 product records"""
        self.products = []
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Toys']
        
        # Generate 5000 products
        for i in range(1, 5001):
            product_id = f'PROD{i:04d}'
            
            # Approximately 9% Electronics (450 products)
            if i <= 450:
                category = 'Electronics'
            else:
                category = random.choice(['Clothing', 'Books', 'Home', 'Toys'])
            
            # 5-10% should have None price
            if random.random() < 0.075:
                price = None
            else:
                price = round(random.uniform(9.99, 999.99), 2)
            
            in_stock = random.choice([True, False])
            
            product = {
                'ProductID': product_id,
                'Category': category,
                'Price': price,
                'InStock': in_stock
            }
            self.products.append(product)
        
        # Shuffle to mix categories
        random.shuffle(self.products)
        
    def scan(self, FilterExpression=None, ExclusiveStartKey=None):
        """
        Simulate DynamoDB scan operation with pagination
        
        Args:
            FilterExpression: dict with attribute filters like {'Category': 'Electronics'}
            ExclusiveStartKey: dict with {'ProductID': 'PROD0100'} to continue from
            
        Returns:
            dict with 'Items' and optionally 'LastEvaluatedKey'
        """
        # Determine start position
        start_index = 0
        if ExclusiveStartKey and 'ProductID' in ExclusiveStartKey:
            # Find the position after the ExclusiveStartKey
            for idx, product in enumerate(self.products):
                if product['ProductID'] == ExclusiveStartKey['ProductID']:
                    start_index = idx + 1
                    break
        
        # Collect items for this page
        page_items = []
        current_index = start_index
        max_items = 100
        
        # Occasionally return empty page with continuation token (5% chance)
        if random.random() < 0.05 and current_index < len(self.products):
            response = {
                'Items': [],
                'LastEvaluatedKey': {'ProductID': self.products[current_index]['ProductID']}
            }
            return response
        
        # Scan through products
        while current_index < len(self.products) and len(page_items) < max_items:
            product = self.products[current_index]
            
            # Apply filter if provided
            if FilterExpression:
                match = True
                for key, value in FilterExpression.items():
                    if product.get(key) != value:
                        match = False
                        break
                if match:
                    page_items.append(product.copy())
            else:
                page_items.append(product.copy())
            
            current_index += 1
        
        # Build response
        response = {'Items': page_items}
        
        # Add LastEvaluatedKey if there are more items
        if current_index < len(self.products):
            response['LastEvaluatedKey'] = {'ProductID': self.products[current_index - 1]['ProductID']}
        
        return response
```

```python
#!/usr/bin/env python3

import json
from dynamodb_simulator import DynamoDBSimulator

def extract_electronics():
    """Extract all Electronics products from the simulated DynamoDB table"""
    
    # Initialize the simulator
    simulator = DynamoDBSimulator()
    
    # Track extracted products
    electronics_products = []
    seen_product_ids = set()
    
    # Pagination variables
    exclusive_start_key = None
    
    while True:
        # Perform scan with filter for Electronics
        scan_params = {
            'FilterExpression': {'Category': 'Electronics'}
        }
        
        if exclusive_start_key:
            scan_params['ExclusiveStartKey'] = exclusive_start_key
        
        # Execute scan
        response = simulator.scan(**scan_params)
        
        # Process items from this page
        items = response.get('Items', [])
        
        for item in items:
            product_id = item['ProductID']
            
            # Avoid duplicates
            if product_id not in seen_product_ids:
                seen_product_ids.add(product_id)
                
                # Extract only ProductID and Price
                product_data = {
                    'id': product_id,
                    'price': item.get('Price')
                }
                electronics_products.append(product_data)
        
        # Check for more pages
        if 'LastEvaluatedKey' in response:
            exclusive_start_key = response['LastEvaluatedKey']
        else:
            # No more pages, we're done
            break
    
    # Prepare output
    output = {
        'count': len(electronics_products),
        'products': electronics_products
    }
    
    # Write to file
    with open('/home/user/electronics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Extracted {len(electronics_products)} Electronics products")
    return output

if __name__ == '__main__':
    extract_electronics()