#!/usr/bin/env python3

import asyncio
import time

# Global variable to track max queue size
max_queue_size = 0

async def producer(queue, duration):
    """Produces messages at 100 messages per second"""
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < duration:
        message = f'Message #{count}'
        await queue.put(message)
        count += 1
        await asyncio.sleep(0.01)  # 100 messages/second
    
    print(f"Producer finished: {count} messages sent")

async def consumer(queue, duration):
    """Consumes messages at 50 messages per second"""
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < duration:
        try:
            message = await asyncio.wait_for(queue.get(), timeout=0.1)
            # Simulate processing
            await asyncio.sleep(0.02)  # 50 messages/second
            count += 1
        except asyncio.TimeoutError:
            continue
    
    print(f"Consumer finished: {count} messages processed")

async def monitor(queue, duration):
    """Monitors queue size every second"""
    global max_queue_size
    start_time = time.time()
    
    while time.time() - start_time < duration:
        current_size = queue.qsize()
        if current_size > max_queue_size:
            max_queue_size = current_size
        print(f"Queue size: {current_size}, Max: {max_queue_size}")
        await asyncio.sleep(1)
    
    print(f"\nFinal Statistics:")
    print(f"Maximum queue size reached: {max_queue_size}")

async def main():
    """Main function to run the message processor"""
    global max_queue_size
    max_queue_size = 0
    
    # Create unbounded queue
    queue = asyncio.Queue()
    
    # Run duration
    duration = 30
    
    print(f"Starting message processor for {duration} seconds...")
    print("Producer rate: 100 msg/sec")
    print("Consumer rate: 50 msg/sec")
    print()
    
    # Create tasks
    producer_task = asyncio.create_task(producer(queue, duration))
    consumer_task = asyncio.create_task(consumer(queue, duration))
    monitor_task = asyncio.create_task(monitor(queue, duration))
    
    # Wait for all tasks to complete
    await asyncio.gather(producer_task, consumer_task, monitor_task)
    
    return max_queue_size

if __name__ == '__main__':
    final_max = asyncio.run(main())
    print(f"\nTest completed. Maximum queue size: {final_max}")