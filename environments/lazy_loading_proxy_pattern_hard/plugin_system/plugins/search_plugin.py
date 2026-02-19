import time

class SearchPlugin:
    def __init__(self):
        self.initialized = False
        self.index = None
    
    def init(self):
        # Simulate expensive index building
        time.sleep(2.5)
        self.initialized = True
        self.index = {}
    
    def search(self, query):
        return f'Search results for: {query}'
    
    def index_document(self, doc_id, content):
        self.index[doc_id] = content
    
    def get_status(self):
        return 'Search plugin ready'