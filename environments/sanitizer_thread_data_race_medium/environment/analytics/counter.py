#!/usr/bin/env python3

import threading
import time


class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        temp = self.count
        time.sleep(0.0001)
        temp += 1
        self.count = temp
    
    def get_count(self):
        return self.count
    
    def reset(self):
        self.count = 0


def worker_thread(counter, num_increments):
    for _ in range(num_increments):
        counter.increment()


if __name__ == '__main__':
    counter = Counter()
    threads = []
    
    for _ in range(10):
        thread = threading.Thread(target=worker_thread, args=(counter, 1000))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print(f"Final count: {counter.get_count()}")