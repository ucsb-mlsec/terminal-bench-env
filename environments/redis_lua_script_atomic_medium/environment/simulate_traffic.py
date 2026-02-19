#!/usr/bin/env python3

import threading
from redis_mock import MockRedis
from broken_counter import PageViewTracker

def simulate_concurrent_traffic():
    # Create a MockRedis instance
    redis_client = MockRedis()
    
    # Initialize the mock Redis with starting counters
    redis_client.set('page:home', 1000)
    redis_client.set('page:products', 500)
    redis_client.set('page:about', 250)
    
    # Create a PageViewTracker instance
    tracker = PageViewTracker(redis_client)
    
    # Function to increment a page view
    def increment_page(page_key, amount=1):
        tracker.increment_view(page_key, amount)
    
    # Create threads for concurrent operations
    threads = []
    
    # 25 threads incrementing 'page:home' by 1
    for _ in range(25):
        thread = threading.Thread(target=increment_page, args=('page:home', 1))
        threads.append(thread)
    
    # 15 threads incrementing 'page:products' by 1
    for _ in range(15):
        thread = threading.Thread(target=increment_page, args=('page:products', 1))
        threads.append(thread)
    
    # 10 threads incrementing 'page:about' by 1
    for _ in range(10):
        thread = threading.Thread(target=increment_page, args=('page:about', 1))
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Retrieve final values from Redis
    home_value = int(redis_client.get('page:home'))
    products_value = int(redis_client.get('page:products'))
    about_value = int(redis_client.get('page:about'))
    
    # Check if values are correct
    expected_home = 1025
    expected_products = 515
    expected_about = 260
    
    is_correct = (home_value == expected_home and 
                  products_value == expected_products and 
                  about_value == expected_about)
    
    correct_str = "yes" if is_correct else "no"
    
    # Format results
    results = f"""page:home={home_value}
page:products={products_value}
page:about={about_value}
correct={correct_str}"""
    
    # Write results to file
    with open('/app/result.txt', 'w') as f:
        f.write(results)
    
    # Print results to console
    print(results)
    print(f"\nExpected: home={expected_home}, products={expected_products}, about={expected_about}")
    print(f"Actual: home={home_value}, products={products_value}, about={about_value}")

if __name__ == '__main__':
    simulate_concurrent_traffic()