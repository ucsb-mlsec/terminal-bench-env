#!/usr/bin/env python3

import grpc
from concurrent import futures
import time
import order_pb2
import order_pb2_grpc


class OrderServiceServicer(order_pb2_grpc.OrderServiceServicer):
    def ProcessOrder(self, request, context):
        # BUG 1: Sleep for 6 seconds - exceeds the 5 second deadline
        time.sleep(6)
        
        # Validate order
        if not request.order_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Order ID is required')
            return order_pb2.OrderResponse()
        
        if request.total_amount <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Total amount must be positive')
            return order_pb2.OrderResponse()
        
        # BUG 2: Blocking operation without checking context cancellation
        for i in range(10):
            time.sleep(0.5)  # Additional delays without checking if context is cancelled
            # Missing context.is_active() check here
        
        # Simulate database storage with delay
        # BUG 3: Another delay without respecting context
        time.sleep(2)
        
        # Process order
        print(f"Processing order {request.order_id} for customer {request.customer_id}")
        print(f"Items: {len(request.items)}, Total: ${request.total_amount}")
        
        # Return success response
        return order_pb2.OrderResponse(
            order_id=request.order_id,
            success=True,
            message=f"Order {request.order_id} processed successfully"
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    order_pb2_grpc.add_OrderServiceServicer_to_server(
        OrderServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    print("Order service starting on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()