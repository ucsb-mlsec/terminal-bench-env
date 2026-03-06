#!/usr/bin/env python3
# DO NOT MODIFY - gRPC client implementation

import grpc
import yaml
import time
import sys
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import generated protobuf files
try:
    import inventory_pb2
    import inventory_pb2_grpc
except ImportError:
    # Create minimal protobuf stubs if not generated
    class inventory_pb2:
        class CheckInventoryRequest:
            def __init__(self, product_id=None, quantity=None):
                self.product_id = product_id
                self.quantity = quantity
        
        class CheckInventoryResponse:
            def __init__(self, available=False, message=""):
                self.available = available
                self.message = message
    
    class inventory_pb2_grpc:
        class InventoryServiceStub:
            def __init__(self, channel):
                self.channel = channel
            
            def CheckInventory(self, request, timeout=None):
                # Simulate gRPC call
                time.sleep(0.1)
                return inventory_pb2.CheckInventoryResponse(available=True, message="OK")


class OrderClient:
    def __init__(self, config_path='/workspace/order_system/config.yaml'):
        """Initialize the order client with configuration."""
        self.config_path = config_path
        self.client_timeout = self._load_config()
        self.channel = None
        self.stub = None
        
    def _load_config(self) -> int:
        """Load client timeout from configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                timeout = config.get('client_timeout', 10)
                logger.info(f"Loaded client_timeout: {timeout} seconds")
                return timeout
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return 10  # Default timeout
    
    def connect(self, server_address='localhost:50051'):
        """Establish connection to the inventory service."""
        self.channel = grpc.insecure_channel(server_address)
        self.stub = inventory_pb2_grpc.InventoryServiceStub(self.channel)
        logger.info(f"Connected to inventory service at {server_address}")
    
    def place_order(self, product_id: str, quantity: int) -> Tuple[bool, str]:
        """
        Place an order by checking inventory availability.
        
        Args:
            product_id: The product identifier
            quantity: Quantity to order
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.stub:
            self.connect()
        
        request = inventory_pb2.CheckInventoryRequest(
            product_id=product_id,
            quantity=quantity
        )
        
        try:
            logger.debug(f"Calling inventory service for product {product_id}, quantity {quantity}")
            response = self.stub.CheckInventory(request, timeout=self.client_timeout)
            
            if response.available:
                logger.info(f"Order placed successfully: {response.message}")
                return True, response.message
            else:
                logger.warning(f"Order failed: {response.message}")
                return False, response.message
                
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                error_msg = f"Request timed out after {self.client_timeout} seconds"
                logger.error(error_msg)
                return False, error_msg
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                error_msg = "Inventory service unavailable"
                logger.error(error_msg)
                return False, error_msg
            else:
                error_msg = f"gRPC error: {e.code()} - {e.details()}"
                logger.error(error_msg)
                return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            logger.info("Channel closed")


if __name__ == "__main__":
    # Test individual order
    client = OrderClient()
    success, message = client.place_order("TEST_PRODUCT_001", 5)
    print(f"Order result: {'SUCCESS' if success else 'FAILED'} - {message}")
    client.close()