#!/usr/bin/env python3

import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold, timeout):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function execution
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            Exception: Any exception raised by the wrapped function
        """
        if self.state == CircuitState.OPEN:
            # Bug 1: Not checking if timeout has elapsed
            # Should check: time.time() - self.last_failure_time >= self.timeout
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenException("Circuit breaker is open")
        
        if self.state == CircuitState.HALF_OPEN:
            # Attempt recovery with one test call
            try:
                result = func(*args, **kwargs)
                # Bug 2: Not resetting failure_count on success
                # Should reset: self.failure_count = 0
                self.state = CircuitState.CLOSED
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                self.state = CircuitState.OPEN
                raise e
        
        # CLOSED state - execute normally
        try:
            result = func(*args, **kwargs)
            # Success - could reset failure count here but not required
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Bug 3: Wrong comparison operator
            # Should be: >= not >
            if self.failure_count > self.failure_threshold:
                self.state = CircuitState.OPEN
            
            raise e