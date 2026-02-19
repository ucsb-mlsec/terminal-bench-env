import time

class APIPlugin:
    def __init__(self):
        self.initialized = False
        self.api_key = None
    
    def init(self):
        time.sleep(2.5)
        self.initialized = True
        self.api_key = 'secret_key_loaded'
    
    def call_api(self, endpoint):
        if not self.initialized:
            raise RuntimeError("Plugin not initialized. Call init() first.")
        return f'API response from: {endpoint}'
    
    def get_status(self):
        return 'API plugin ready'