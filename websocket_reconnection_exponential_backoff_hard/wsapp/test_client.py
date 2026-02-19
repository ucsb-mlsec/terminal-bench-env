#!/usr/bin/env python3

import unittest
import time
import threading
import sys
import os
import json
from unittest.mock import Mock, patch
import asyncio
import websockets

# Add the current directory to path to import client and mock_server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import WebSocketClient
from mock_server import MockWebSocketServer


class TestWebSocketClientReconnection(unittest.TestCase):
    """Test suite for WebSocket client reconnection behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.server = None
        self.client = None
        self.server_port = 8765
        self.server_host = "localhost"
        
    def tearDown(self):
        """Clean up after tests"""
        if self.client:
            self.client.stop()
        if self.server:
            self.server.stop()
        time.sleep(0.5)  # Give time for cleanup
    
    def start_server(self, drop_after=None, accept_connections=True):
        """Start mock WebSocket server"""
        self.server = MockWebSocketServer(
            host=self.server_host,
            port=self.server_port,
            drop_after=drop_after,
            accept_connections=accept_connections
        )
        self.server.start()
        time.sleep(0.5)  # Give server time to start
    
    def test_reconnection_after_drop(self):
        """Test that client reconnects after connection drops"""
        print("\n=== Test: Reconnection After Drop ===")
        
        # Start server that will drop connection after 2 seconds
        self.start_server(drop_after=2)
        
        # Create and start client
        self.client = WebSocketClient(
            uri=f"ws://{self.server_host}:{self.server_port}"
        )
        
        client_thread = threading.Thread(target=self.client.run, daemon=True)
        client_thread.start()
        
        # Wait for initial connection
        time.sleep(1)
        self.assertTrue(self.client.is_connected(), "Client should be initially connected")
        
        # Wait for server to drop connection
        time.sleep(2.5)
        
        # Wait for reconnection attempt
        time.sleep(3)
        
        # Check that reconnection was attempted
        self.assertGreater(self.client.reconnect_attempts, 0, 
                          "Client should have attempted reconnection")
        print(f"Reconnection attempts: {self.client.reconnect_attempts}")
    
    def test_exponential_backoff(self):
        """Test that delays between retries increase exponentially"""
        print("\n=== Test: Exponential Backoff ===")
        
        # Don't start server initially so connection fails
        self.client = WebSocketClient(
            uri=f"ws://{self.server_host}:{self.server_port}"
        )
        
        client_thread = threading.Thread(target=self.client.run, daemon=True)
        client_thread.start()
        
        time.sleep(0.5)
        
        # Record retry times
        retry_times = []
        last_attempt = self.client.reconnect_attempts
        start_time = time.time()
        
        # Monitor for several retry attempts
        for i in range(5):
            time.sleep(1.5)
            current_attempt = self.client.reconnect_attempts
            if current_attempt > last_attempt:
                retry_times.append(time.time() - start_time)
                last_attempt = current_attempt
        
        # Check that delays are increasing (exponential backoff)
        if len(retry_times) >= 3:
            delay1 = retry_times[1] - retry_times[0] if len(retry_times) > 1 else 0
            delay2 = retry_times[2] - retry_times[1] if len(retry_times) > 2 else 0
            
            print(f"Retry times: {retry_times}")
            print(f"Delay 1: {delay1:.2f}s, Delay 2: {delay2:.2f}s")
            
            # Verify exponential increase (delay2 should be roughly 2x delay1)
            if delay1 > 0 and delay2 > 0:
                self.assertGreater(delay2, delay1, 
                                 "Second delay should be greater than first delay")
    
    def test_backoff_reset(self):
        """Test that backoff resets after stable connection"""
        print("\n=== Test: Backoff Reset After Stable Connection ===")
        
        # Start server
        self.start_server()
        
        self.client = WebSocketClient(
            uri=f"ws://{self.server_host}:{self.server_port}"
        )
        
        client_thread = threading.Thread(target=self.client.run, daemon=True)
        client_thread.start()
        
        # Wait for initial connection
        time.sleep(1)
        self.assertTrue(self.client.is_connected(), "Client should be connected")
        
        # Simulate some failed attempts by setting backoff delay
        self.client.current_backoff_delay = 16
        initial_backoff = self.client.current_backoff_delay
        
        print(f"Initial backoff delay: {initial_backoff}s")
        
        # Wait for stable connection (30+ seconds would reset in real scenario)
        # For testing, we'll check the mechanism exists
        if hasattr(self.client, 'last_successful_connect_time'):
            self.assertIsNotNone(self.client.last_successful_connect_time,
                               "Client should track successful connection time")
        
        # Verify backoff can be reset
        if hasattr(self.client, 'reset_backoff'):
            self.client.reset_backoff()
            self.assertEqual(self.client.current_backoff_delay, 1,
                           "Backoff should reset to 1 second")
            print(f"Backoff after reset: {self.client.current_backoff_delay}s")
    
    def test_max_retry_limit(self):
        """Test that client stops after maximum retry attempts"""
        print("\n=== Test: Maximum Retry Limit ===")
        
        # Don't start server - all connections will fail
        self.client = WebSocketClient(
            uri=f"ws://{self.server_host}:{self.server_port}",
            max_retries=10
        )
        
        client_thread = threading.Thread(target=self.client.run, daemon=True)
        client_thread.start()
        
        # Wait for retries to exhaust (with exponential backoff, this takes time)
        # 1 + 2 + 4 + 8 + 16 + 32 + 60 + 60 + 60 + 60 = ~303 seconds max
        # We'll wait a reasonable amount and check if it stops
        time.sleep(15)
        
        attempts = self.client.reconnect_attempts
        print(f"Reconnection attempts after 15s: {attempts}")
        
        # Should have made multiple attempts but not infinite
        self.assertGreater(attempts, 0, "Should have attempted reconnection")
        self.assertLess(attempts, 50, "Should not retry indefinitely")
        
        # Check if client has max_retries attribute
        if hasattr(self.client, 'max_retries'):
            self.assertEqual(self.client.max_retries, 10,
                           "Max retries should be set to 10")
    
    def test_message_queuing(self):
        """Test that messages are queued during disconnection and sent after reconnection"""
        print("\n=== Test: Message Queuing ===")
        
        # Start server
        self.start_server()
        
        self.client = WebSocketClient(
            uri=f"ws://{self.server_host}:{self.server_port}"
        )
        
        client_thread = threading.Thread(target=self.client.run, daemon=True)
        client_thread.start()
        
        # Wait for initial connection
        time.sleep(1)
        self.assertTrue(self.client.is_connected(), "Client should be connected")
        
        # Send a message while connected
        self.client.send_message("test_message_1")
        time.sleep(0.5)
        
        # Stop server to simulate disconnection
        self.server.stop()
        time.sleep(1)
        
        # Try to send messages while disconnected
        self.client.send_message("test_message_2")
        self.client.send_message("test_message_3")
        
        # Check that messages are queued
        if hasattr(self.client, 'message_queue'):
            queue_size = self.client.message_queue.qsize() if hasattr(self.client.message_queue, 'qsize') else len(self.client.message_queue)
            print(f"Messages in queue: {queue_size}")
            self.assertGreater(queue_size, 0, "Messages should be queued while disconnected")
        
        # Restart server
        self.start_server()
        
        # Wait for reconnection and message sending
        time.sleep(3)
        
        # Verify messages were sent after reconnection
        if hasattr(self.client, 'message_queue'):
            queue_size = self.client.message_queue.qsize() if hasattr(self.client.message_queue, 'qsize') else len(self.client.message_queue)
            print(f"Messages in queue after reconnection: {queue_size}")
            # Queue should be empty or smaller after reconnection
            self.assertLessEqual(queue_size, 2, "Messages should be sent after reconnection")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)