# DO NOT MODIFY - gRPC server implementation

import grpc
from concurrent import futures
import time
import yaml
import random
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import generated protobuf files
try:
    import inventory_pb2
    import inventory_pb2_grpc
except ImportError:
    # Mock implementations for protobuf if not generated
    class MockMessage:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class inventory_pb2:
        class InventoryRequest(MockMessage):
            pass
        
        class InventoryResponse(MockMessage):
            pass
    
    class inventory_pb2_grpc:
        @staticmethod
        def add_InventoryServiceServicer_to_server(servicer, server):
            pass


class InventoryServicer:
    """gRPC servicer for inventory service"""
    
    def __init__(self, server_timeout):
        self.server_timeout = server_timeout
        logger.info(f"InventoryServicer initialized with server_timeout={server_timeout}s")
    
    def CheckInventory(self, request, context):
        """
        Check inventory availability for a product
        Simulates variable processing time based on system load
        """
        try:
            product_id = getattr(request, 'product_id', 'unknown')
            quantity = getattr(request, 'quantity', 0)
            
            # Simulate variable processing time (1-12 seconds)
            # Distribution: 70% complete in 1-6s, 20% in 7-10s, 10% in 11-12s
            rand_val = random.random()
            if rand_val < 0.70:
                processing_time = random.uniform(1, 6)
            elif rand_val < 0.90:
                processing_time = random.uniform(7, 10)
            else:
                processing_time = random.uniform(11, 12)
            
            logger.info(f"Processing inventory check for product_id={product_id}, "
                       f"quantity={quantity}, estimated_time={processing_time:.2f}s")
            
            # Simulate processing with timeout awareness
            start_time = time.time()
            elapsed = 0
            
            while elapsed < processing_time:
                # Check if request is still active
                if context.is_active():
                    sleep_time = min(0.5, processing_time - elapsed)
                    time.sleep(sleep_time)
                    elapsed = time.time() - start_time
                else:
                    logger.warning(f"Request cancelled or timed out after {elapsed:.2f}s")
                    context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                    context.set_details('Request deadline exceeded on server')
                    return inventory_pb2.InventoryResponse(available=False, message="Timeout")
            
            # Simulate inventory availability (90% available)
            available = random.random() < 0.90
            
            logger.info(f"Inventory check completed in {elapsed:.2f}s, available={available}")
            
            return inventory_pb2.InventoryResponse(
                available=available,
                message=f"Processed in {elapsed:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error processing inventory check: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error: {str(e)}')
            return inventory_pb2.InventoryResponse(available=False, message="Error")


def load_config():
    """Load configuration from config.yaml"""
    config_path = '/workspace/order_system/config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {'server_timeout': 15}  # Default value
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {'server_timeout': 15}  # Default value


def serve():
    """Start the gRPC server"""
    # Load configuration
    config = load_config()
    server_timeout = config.get('server_timeout', 15)
    
    logger.info(f"Starting inventory service with server_timeout={server_timeout}s")
    
    # Create gRPC server with thread pool
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=50),
        options=[
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ]
    )
    
    # Add servicer to server
    inventory_servicer = InventoryServicer(server_timeout)
    inventory_pb2_grpc.add_InventoryServiceServicer_to_server(inventory_servicer, server)
    
    # Bind to port
    port = '50051'
    server.add_insecure_port(f'[::]:{port}')
    
    # Start server
    server.start()
    logger.info(f"Inventory service listening on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down inventory service...")
        server.stop(grace=5)


if __name__ == '__main__':
    serve()