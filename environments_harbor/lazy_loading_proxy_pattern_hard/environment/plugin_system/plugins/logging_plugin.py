import time

class LoggingPlugin:
    def __init__(self):
        self.initialized = False
        self.log_handlers = None
    
    def init(self):
        # Simulate expensive log file parsing
        time.sleep(2.5)
        self.initialized = True
        self.log_handlers = []
    
    def log(self, message, level='INFO'):
        self.log_handlers.append({'message': message, 'level': level})
    
    def get_logs(self):
        return self.log_handlers
    
    def get_status(self):
        return 'Logging plugin ready'