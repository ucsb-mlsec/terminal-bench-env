#!/usr/bin/env python3

import threading
import time
import uuid
from typing import Dict

class QueryService:
    def __init__(self):
        self.queries: Dict[str, dict] = {}
        self.lock = threading.Lock()
    
    def submit_query(self, sql: str) -> str:
        """Submit a query for execution. Returns a query_id immediately."""
        query_id = str(uuid.uuid4())
        
        with self.lock:
            self.queries[query_id] = {
                'sql': sql,
                'start_time': time.time(),
                'complete': False,
                'result': None
            }
        
        # Start execution in background thread
        thread = threading.Thread(target=self._execute_query, args=(query_id,))
        thread.daemon = True
        thread.start()
        
        return query_id
    
    def _execute_query(self, query_id: str):
        """Internal method to execute query after 4 seconds."""
        time.sleep(4)  # Simulate 4-second execution
        
        with self.lock:
            if query_id in self.queries:
                self.queries[query_id]['complete'] = True
                self.queries[query_id]['result'] = {
                    'status': 'success',
                    'rows': 100  # Mock row count
                }
    
    def is_complete(self, query_id: str) -> bool:
        """Check if a query has finished executing."""
        with self.lock:
            if query_id not in self.queries:
                raise ValueError(f"Query ID {query_id} not found")
            return self.queries[query_id]['complete']
    
    def get_result(self, query_id: str) -> dict:
        """Retrieve the result of a completed query."""
        with self.lock:
            if query_id not in self.queries:
                raise ValueError(f"Query ID {query_id} not found")
            
            query_data = self.queries[query_id]
            if not query_data['complete']:
                raise RuntimeError(f"Query {query_id} is not yet complete")
            
            return query_data['result']