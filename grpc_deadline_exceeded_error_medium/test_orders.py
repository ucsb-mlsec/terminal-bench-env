#!/usr/bin/env python3

import yaml
import sys
import random
import time

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('/workspace/order_system/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config.get('client_timeout', 10), config.get('server_timeout', 15)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def generate_processing_time():
    """Generate a random processing time based on the distribution"""
    rand = random.random()
    
    if rand < 0.60:  # 60% of orders: 1-6 seconds
        return random.uniform(1, 6)
    elif rand < 0.85:  # 25% of orders: 8-10 seconds
        return random.uniform(8, 10)
    elif rand < 0.95:  # 10% of orders: 11-12 seconds
        return random.uniform(11, 12)
    else:  # 5% of orders: 21-25 seconds (stuck requests)
        return random.uniform(21, 25)

def simulate_order(client_timeout, server_timeout, processing_time):
    """Simulate an order processing operation"""
    # The order fails if processing time exceeds client timeout
    if processing_time <= client_timeout:
        return True  # SUCCESS
    else:
        return False  # FAILED

def main():
    print("Testing 100 orders...")
    
    # Load timeout configuration
    client_timeout, server_timeout = load_config()
    
    # Track results
    total_orders = 100
    successful_orders = 0
    legitimate_orders = 0
    successful_legitimate = 0
    
    # Simulate orders
    for i in range(total_orders):
        processing_time = generate_processing_time()
        success = simulate_order(client_timeout, server_timeout, processing_time)
        
        if success:
            successful_orders += 1
        
        # Track legitimate operations (those that complete within 15 seconds)
        if processing_time <= 15:
            legitimate_orders += 1
            if success:
                successful_legitimate += 1
    
    # Calculate success rate
    success_rate = (successful_orders / total_orders) * 100
    
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Verify that legitimate operations pass at >= 95% rate
    if legitimate_orders > 0:
        legitimate_success_rate = (successful_legitimate / legitimate_orders) * 100
        # print(f"Legitimate Operations Success Rate: {legitimate_success_rate:.1f}%")
    
    # Exit with appropriate code
    if success_rate >= 95.0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()