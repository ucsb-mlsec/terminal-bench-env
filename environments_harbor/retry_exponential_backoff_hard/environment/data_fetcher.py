#!/usr/bin/env python3

import requests
import logging
import time
import json
import random
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetryState:
    """Tracks retry state for operations, persists to disk"""
    
    def __init__(self, state_file: str = "/tmp/retry_state.json"):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load retry state from disk"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load retry state: {e}")
        return {}
    
    def _save_state(self):
        """Persist retry state to disk"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"Failed to save retry state: {e}")
    
    def get_attempt_count(self, operation_id: str) -> int:
        """Get current attempt count for an operation"""
        return self.state.get(operation_id, {}).get('attempts', 0)
    
    def record_attempt(self, operation_id: str):
        """Record a retry attempt"""
        if operation_id not in self.state:
            self.state[operation_id] = {'attempts': 0, 'first_attempt': datetime.now().isoformat()}
        self.state[operation_id]['attempts'] += 1
        self.state[operation_id]['last_attempt'] = datetime.now().isoformat()
        self._save_state()
    
    def clear_operation(self, operation_id: str):
        """Clear retry state for completed operation"""
        if operation_id in self.state:
            del self.state[operation_id]
            self._save_state()
    
    def is_expired(self, operation_id: str, max_duration_seconds: int = 300) -> bool:
        """Check if operation has exceeded maximum retry duration"""
        if operation_id not in self.state:
            return False
        first_attempt = datetime.fromisoformat(self.state[operation_id]['first_attempt'])
        return (datetime.now() - first_attempt).total_seconds() > max_duration_seconds


class CircuitBreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent overwhelming failing services"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN, rejecting request")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.timeout_seconds
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            logger.info("Circuit breaker closing after successful test")
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class DataFetcher:
    """Fetches data from external API with robust retry mechanism"""
    
    def __init__(self, api_base_url: str, max_attempts: int = 5, 
                 base_delay_ms: int = 1000, max_delay_ms: int = 30000):
        self.api_base_url = api_base_url
        self.max_attempts = max_attempts
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.retry_state = RetryState()
        self.circuit_breaker = CircuitBreaker()
        self.session = requests.Session()
        
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        # Exponential backoff: base_delay * 2^attempt
        delay_ms = min(self.base_delay_ms * (2 ** attempt), self.max_delay_ms)
        
        # Add jitter (random ±25% variation) to prevent thundering herd
        jitter = random.uniform(-0.25, 0.25)
        delay_ms = delay_ms * (1 + jitter)
        
        return delay_ms / 1000.0  # Convert to seconds
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make actual HTTP request"""
        url = f"{self.api_base_url}/{endpoint}"
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def fetch_data(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Fetch data with exponential backoff retry mechanism"""
        operation_id = f"{endpoint}_{hash(str(params))}"
        
        # Check if operation has exceeded maximum duration
        if self.retry_state.is_expired(operation_id):
            logger.error(f"Operation {operation_id} exceeded maximum retry duration")
            self.retry_state.clear_operation(operation_id)
            return None
        
        attempt_count = self.retry_state.get_attempt_count(operation_id)
        
        for attempt in range(attempt_count, self.max_attempts):
            try:
                # Record attempt
                self.retry_state.record_attempt(operation_id)
                
                logger.info(f"Attempt {attempt + 1}/{self.max_attempts} for {endpoint}")
                
                # Use circuit breaker to protect against cascading failures
                result = self.circuit_breaker.call(self._make_request, endpoint, params)
                
                # Success - clear retry state
                self.retry_state.clear_operation(operation_id)
                logger.info(f"Successfully fetched data from {endpoint}")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {endpoint}: {str(e)}")
                
                if attempt < self.max_attempts - 1:
                    # Calculate backoff delay with exponential increase and jitter
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Backing off for {delay:.2f} seconds before retry")
                    time.sleep(delay)
                else:
                    # All retries exhausted
                    logger.error(f"All {self.max_attempts} attempts failed for {endpoint}")
                    self.retry_state.clear_operation(operation_id)
        
        return None


class JobProcessor:
    """Coordinates multiple fetch operations with graceful degradation"""
    
    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher
    
    def process_batch(self, endpoints: list) -> Dict[str, Any]:
        """Process multiple endpoints, returning partial results on failure"""
        results = {}
        successful = 0
        failed = 0
        
        for endpoint in endpoints:
            try:
                data = self.fetcher.fetch_data(endpoint)
                if data is not None:
                    results[endpoint] = data
                    successful += 1
                else:
                    results[endpoint] = None
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to process {endpoint}: {e}")
                results[endpoint] = None
                failed += 1
        
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        
        return {
            'results': results,
            'successful': successful,
            'failed': failed,
            'partial': failed > 0
        }


class HealthMonitor:
    """Tracks service availability and health metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0,
            'circuit_breaker_trips': 0
        }
    
    def record_request(self, success: bool, retry_count: int = 0):
        """Record request metrics"""
        self.metrics['total_requests'] += 1
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        self.metrics['retry_attempts'] += retry_count
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        total = self.metrics['total_requests']
        if total == 0:
            success_rate = 0
        else:
            success_rate = self.metrics['successful_requests'] / total * 100
        
        return {
            'status': 'healthy' if success_rate > 80 else 'degraded',
            'success_rate': f"{success_rate:.2f}%",
            'metrics': self.metrics
        }


def main():
    """Main execution and testing"""
    # Configuration
    MAX_ATTEMPTS = 5
    BASE_DELAY_MS = 1000
    MAX_DELAY_MS = 30000
    
    # Create solution.json
    solution_data = {
        "max_total_attempts": MAX_ATTEMPTS,
        "base_delay_ms": BASE_DELAY_MS,
        "max_delay_ms": MAX_DELAY_MS,
        "implementation_file": os.path.abspath(__file__)
    }
    
    with open('/tmp/solution.json', 'w') as f:
        json.dump(solution_data, f, indent=2)
    
    logger.info("Solution configuration saved to /tmp/solution.json")
    
    # Example usage
    fetcher = DataFetcher(
        api_base_url="https://api.example.com",
        max_attempts=MAX_ATTEMPTS,
        base_delay_ms=BASE_DELAY_MS,
        max_delay_ms=MAX_DELAY_MS
    )
    
    processor = JobProcessor(fetcher)
    monitor = HealthMonitor()
    
    logger.info("System initialized with robust retry mechanism")
    logger.info(f"Max attempts: {MAX_ATTEMPTS}, Base delay: {BASE_DELAY_MS}ms, Max delay: {MAX_DELAY_MS}ms")


if __name__ == "__main__":
    main()