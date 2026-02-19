#!/usr/bin/env python3

import logging
import time
import requests
import threading
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class HealthMonitor:
    """
    Basic health monitor that tracks service availability.
    Problematic implementation with no backoff, synchronized checks,
    and non-persistent state.
    """
    
    def __init__(self, check_interval: int = 5):
        """
        Initialize health tracking state.
        
        Args:
            check_interval: Fixed interval between health checks in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.check_interval = check_interval
        self.service_status: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.running = True
        
    def check_health(self, service_name: str) -> bool:
        """
        Ping a service endpoint to check health.
        Immediate check with no backoff on failures.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Assume service endpoint follows pattern
            endpoint = f"http://{service_name}/health"
            response = requests.get(endpoint, timeout=2)
            
            if response.status_code == 200:
                self.logger.info(f"Health check passed for {service_name}")
                return True
            else:
                self.logger.warning(
                    f"Health check failed for {service_name}: "
                    f"status {response.status_code}"
                )
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"Health check error for {service_name}: {e}")
            return False
    
    def record_failure(self, service_name: str) -> None:
        """
        Record a failed health check.
        Simple counter increment with no sophistication.
        
        Args:
            service_name: Name of the service that failed
        """
        with self.lock:
            if service_name not in self.service_status:
                self.service_status[service_name] = {
                    'failure_count': 0,
                    'success_count': 0,
                    'last_check': None,
                    'status': 'unknown'
                }
            
            self.service_status[service_name]['failure_count'] += 1
            self.service_status[service_name]['last_check'] = datetime.now()
            self.service_status[service_name]['status'] = 'unhealthy'
            
            self.logger.warning(
                f"Recorded failure for {service_name}. "
                f"Total failures: {self.service_status[service_name]['failure_count']}"
            )
    
    def record_success(self, service_name: str) -> None:
        """
        Record a successful health check.
        
        Args:
            service_name: Name of the service that succeeded
        """
        with self.lock:
            if service_name not in self.service_status:
                self.service_status[service_name] = {
                    'failure_count': 0,
                    'success_count': 0,
                    'last_check': None,
                    'status': 'unknown'
                }
            
            self.service_status[service_name]['success_count'] += 1
            self.service_status[service_name]['last_check'] = datetime.now()
            self.service_status[service_name]['status'] = 'healthy'
            
            self.logger.info(
                f"Recorded success for {service_name}. "
                f"Total successes: {self.service_status[service_name]['success_count']}"
            )
    
    def get_service_status(self, service_name: str) -> Optional[dict]:
        """
        Get current health status for a service.
        
        Args:
            service_name: Name of the service to query
            
        Returns:
            Dictionary with service health status or None if not tracked
        """
        with self.lock:
            return self.service_status.get(service_name)
    
    def _monitor_service(self, service_name: str) -> None:
        """
        Continuous monitoring loop for a service.
        Problematic: no jitter, fixed intervals, synchronized checks.
        
        Args:
            service_name: Name of service to monitor
        """
        self.logger.info(f"Starting health monitoring for {service_name}")
        
        while self.running:
            # Immediate check with no backoff
            is_healthy = self.check_health(service_name)
            
            if is_healthy:
                self.record_success(service_name)
            else:
                self.record_failure(service_name)
            
            # Fixed interval wait - no jitter, causes synchronized checks
            time.sleep(self.check_interval)
    
    def start_monitoring(self, service_name: str) -> None:
        """
        Start monitoring a service in a background thread.
        
        Args:
            service_name: Name of service to monitor
        """
        if service_name in self.monitoring_threads:
            self.logger.warning(f"Already monitoring {service_name}")
            return
        
        thread = threading.Thread(
            target=self._monitor_service,
            args=(service_name,),
            daemon=True
        )
        thread.start()
        self.monitoring_threads[service_name] = thread
        self.logger.info(f"Started monitoring thread for {service_name}")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring threads."""
        self.logger.info("Stopping all health monitoring")
        self.running = False
        
        for service_name, thread in self.monitoring_threads.items():
            thread.join(timeout=10)
            self.logger.info(f"Stopped monitoring {service_name}")
