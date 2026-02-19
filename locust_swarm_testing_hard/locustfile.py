#!/usr/bin/env python3

# Load test definition for microservices application
# This file defines user behavior patterns for stress testing

from locust import HttpUser, task, between
import json


class MicroserviceUser(HttpUser):
    """
    Simulates user behavior for the microservices application.
    Users will perform various API operations with random wait times.
    """
    
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)
    
    @task(3)
    def get_users(self):
        """
        Fetch user list from the API.
        This is weighted higher (3) as it's a common operation.
        """
        response = self.client.get("/api/users")
        if response.status_code == 200:
            # Simulate processing the response
            users = response.json()
    
    @task(2)
    def get_products(self):
        """
        Retrieve product catalog.
        Weighted at 2 for moderate frequency.
        """
        response = self.client.get("/api/products")
        if response.status_code == 200:
            products = response.json()
    
    @task(1)
    def create_order(self):
        """
        Create a new order via POST request.
        Weighted at 1 as it's less frequent than reads.
        """
        order_data = {
            "user_id": 123,
            "product_id": 456,
            "quantity": 2
        }
        headers = {"Content-Type": "application/json"}
        response = self.client.post(
            "/api/orders",
            data=json.dumps(order_data),
            headers=headers
        )
        
    @task(2)
    def get_orders(self):
        """
        Fetch existing orders.
        Another read operation with weight 2.
        """
        response = self.client.get("/api/orders")