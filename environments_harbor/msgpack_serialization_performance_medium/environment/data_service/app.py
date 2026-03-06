#!/usr/bin/env python3

import json
import time
import random
import datetime

def generate_transaction_data(count=1000):
    """
    Generate sample transaction data with edge cases.
    
    Args:
        count: Number of transactions to generate
        
    Returns:
        List of transaction dictionaries
    """
    transactions = []
    currencies = ['USD', 'EUR', 'GBP']
    statuses = ['completed', 'pending', 'failed']
    device_types = ['mobile', 'desktop', 'tablet', '']
    locations = ['US', 'UK', 'DE', 'FR', 'JP', '']
    
    for i in range(count):
        # Generate transaction with various data types
        transaction = {
            'transaction_id': f'TXN-{str(i).zfill(5)}',
            'customer_id': f'CUST-{str(random.randint(1, 10000)).zfill(5)}',
            'timestamp': (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 365))).isoformat(),
            'amount': round(random.uniform(10.0, 5000.0), 2),
            'currency': random.choice(currencies),
            'status': random.choice(statuses),
            'description': None if i % 10 == 0 else f'Transaction for order #{random.randint(1000, 9999)}',
            'metadata': {
                'ip_address': f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
                'device_type': random.choice(device_types),
                'location': random.choice(locations)
            }
        }
        
        # Add some edge cases with large numbers
        if i % 50 == 0:
            transaction['amount'] = round(random.uniform(50000.0, 999999.99), 2)
            
        transactions.append(transaction)
    
    return transactions

def serialize_json(data):
    """
    Serialize data using JSON format.
    
    Args:
        data: Data to serialize
        
    Returns:
        Serialized JSON string
    """
    return json.dumps(data)

def deserialize_json(serialized_data):
    """
    Deserialize JSON string back to Python objects.
    
    Args:
        serialized_data: JSON string to deserialize
        
    Returns:
        Deserialized Python data structure
    """
    return json.loads(serialized_data)

def benchmark_json(data, iterations=100):
    """
    Benchmark JSON serialization performance.
    
    Args:
        data: Data to serialize
        iterations: Number of iterations to run
        
    Returns:
        Tuple of (average_time_ms, serialized_size_bytes)
    """
    total_time = 0
    serialized_data = None
    
    for _ in range(iterations):
        start_time = time.time()
        serialized_data = serialize_json(data)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    average_time_ms = (total_time / iterations) * 1000
    size_bytes = len(serialized_data.encode('utf-8'))
    
    return average_time_ms, size_bytes

if __name__ == '__main__':
    print("Generating transaction data...")
    transactions = generate_transaction_data(1000)
    
    print("Benchmarking JSON serialization...")
    json_time, json_size = benchmark_json(transactions, iterations=100)
    
    print(f"\nJSON Serialization Results:")
    print(f"Average time: {json_time:.2f} ms")
    print(f"Serialized size: {json_size} bytes")
    
    # Verify data integrity
    serialized = serialize_json(transactions)
    deserialized = deserialize_json(serialized)
    print(f"Data integrity check: {'PASSED' if len(deserialized) == len(transactions) else 'FAILED'}")