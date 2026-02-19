#!/usr/bin/env python3

from pymemcache.client.hash import HashClient
import time

def process_batches():
    # Connection pool with TOO SMALL max_pool_size - this is the bug
    client = HashClient(
        [('localhost', 11211)],
        max_pool_size=2,  # TOO SMALL - causes pool exhaustion
        connect_timeout=1,
        timeout=1
    )
    
    try:
        for i in range(100):
            # Generate batch data
            data = f'batch_{i}_data_{time.time()}'
            key = f'batch_{i}'
            
            # Store in memcached
            # BUG: Connections are acquired but not properly released
            try:
                client.set(key, data)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                raise
            
            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f'Processed batch {i + 1}/100')
        
        print('SUCCESS: All batches processed')
        
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
    finally:
        # Even the cleanup is insufficient due to pool exhaustion
        pass

if __name__ == '__main__':
    process_batches()