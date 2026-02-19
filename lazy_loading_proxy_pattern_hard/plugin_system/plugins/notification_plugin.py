import time

class NotificationPlugin:
    def __init__(self):
        pass
    
    def init(self):
        time.sleep(2)
        self.initialized = True
        self.channels = []
    
    def send_notification(self, message, channel='default'):
        return f'Sent: {message} to {channel}'
    
    def get_channels(self):
        return self.channels
    
    def get_status(self):
        return 'Notification plugin ready'