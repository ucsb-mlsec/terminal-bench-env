#!/usr/bin/env python3

import logging
import concurrent.futures
import time
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class JobProcessor:
    """
    Coordinates multiple data fetch operations from various endpoints.
    Currently has no proper retry coordination or error handling.
    """
    
    def __init__(self, data_fetcher):
        """
        Initialize the job processor with a data fetcher instance.
        
        Args:
            data_fetcher: Instance of DataFetcher to use for API calls
        """
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_workers = 10
        
    def process_job(self, job_id: str, endpoints_list: List[str]) -> Dict[str, Any]:
        """
        Process a job by fetching data from multiple endpoints in parallel.
        
        Args:
            job_id: Unique identifier for this job
            endpoints_list: List of endpoint URLs to fetch data from
            
        Returns:
            Dictionary mapping endpoint names to their fetched data
        """
        self.logger.info(f"Starting job {job_id} with {len(endpoints_list)} endpoints")
        start_time = time.time()
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all fetch tasks
            future_to_endpoint = {
                executor.submit(self._fetch_endpoint_data, endpoint): endpoint
                for endpoint in endpoints_list
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_endpoint):
                endpoint = future_to_endpoint[future]
                try:
                    # This will raise an exception if the fetch failed
                    data = future.result()
                    results[endpoint] = data
                    self.logger.info(f"Successfully fetched data from {endpoint}")
                except Exception as e:
                    # Poor error handling - just log and continue
                    self.logger.error(f"Failed to fetch from {endpoint}: {e}")
                    # Assumes failure is acceptable, no retry logic
                    results[endpoint] = None
        
        elapsed = time.time() - start_time
        self.logger.info(f"Job {job_id} completed in {elapsed:.2f}s")
        
        return results
    
    def _fetch_endpoint_data(self, endpoint: str) -> Any:
        """
        Fetch data from a single endpoint using the data fetcher.
        
        Args:
            endpoint: The endpoint URL to fetch from
            
        Returns:
            Data returned from the endpoint
        """
        # Direct call to fetcher with no retry coordination
        return self.data_fetcher.fetch(endpoint)