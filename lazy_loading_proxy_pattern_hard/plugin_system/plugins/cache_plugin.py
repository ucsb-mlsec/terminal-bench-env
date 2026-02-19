import time


class CachePlugin:
    def __init__(self):
        self.initialized = False
        self.cache = None
    
    def init(self):
        """Simulate expensive cache initialization by sleeping for 2 seconds."""
        time.sleep(2)
        self.initialized = True
        self.cache = {}
    
    def get(self, key):
        """Get a value from the cache."""
        return self.cache.get(key)
    
    def set(self, key, value):
        """Set a value in the cache."""
        self.cache[key] = value
    
    def get_status(self):
        """Return the plugin status."""
        return 'Cache plugin ready'