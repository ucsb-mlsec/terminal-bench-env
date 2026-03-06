#!/usr/bin/env python3

import time


class DatabasePlugin:
    def __init__(self):
        """Initialize the plugin without expensive operations."""
        self.initialized = False
        self.connection = None
    
    def init(self):
        """Perform expensive initialization (simulates database connection)."""
        print("DatabasePlugin: Starting expensive initialization...")
        time.sleep(2.5)  # Simulate expensive database connection
        self.initialized = True
        self.connection = 'db_connection_object'
        print("DatabasePlugin: Initialization complete")
    
    def query(self, sql):
        """Execute a query - requires initialization."""
        if not self.initialized:
            raise RuntimeError("DatabasePlugin not initialized. Call init() first.")
        return f'Result for: {sql}'
    
    def get_status(self):
        """Get plugin status."""
        return 'Database plugin ready'