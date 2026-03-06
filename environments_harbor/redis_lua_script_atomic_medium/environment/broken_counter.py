#!/usr/bin/env python3

class PageViewTracker:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def increment_view(self, page_key, amount):
        # Fixed implementation: use atomic Redis INCRBY operation
        # This ensures the read-modify-write operation is atomic
        return self.redis.incrby(page_key, amount)