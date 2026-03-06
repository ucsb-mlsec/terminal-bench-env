import time
import threading

class TokenBucket:
    def __init__(self, rate=10, capacity=20):
        self.rate = rate
        self.capacity = capacity
        self.tokens = 0  # BUG 1: Should start with full capacity
        self.last_refill = time.time()
        # BUG 2: Missing thread lock for concurrent access
        
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        
        # BUG 3: Wrong time unit - multiplying by 1000 treats milliseconds as seconds
        new_tokens = elapsed * self.rate * 1000
        self.tokens = self.tokens + new_tokens
        
        # BUG 4: Not capping tokens at capacity
        # self.tokens = min(self.tokens, self.capacity)
        
        self.last_refill = now
        
    def consume(self, tokens=1):
        self._refill()
        
        # BUG 5: Wrong comparison operator - should be >= not >
        if self.tokens > tokens:
            self.tokens -= tokens
            return True
        else:
            return False