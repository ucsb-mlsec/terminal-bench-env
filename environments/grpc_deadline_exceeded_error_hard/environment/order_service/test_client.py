#!/usr/bin/env python3

import grpc
import order_pb2
import order_pb2_grpc
import time


def run_test():
    """Test the order processing service with 10 requests."""
    # Create a gRPC channel to the server
    channel = grpc.insecure_channel('localhost:50051')
    stub = order_pb2_grpc.OrderServiceStub(channel)
    
    success_count = 0
    timeout_count = 0
    
    print("Starting order processing test with 10 requests...\n")
    
    # Send 10 order requests
    for i in range(10):
        order_id = f'ORD-{i+1:03d}'
        customer_id = f'CUST-{i+1}'
        items = ['item1', 'item2']
        total_amount = 100.0 + (i * 10)
        
        # Create the order request
        request = order_pb2.OrderRequest(
            order_id=order_id,
            customer_id=customer_id,
            items=items,
            total_amount=total_amount
        )
        
        try:
            # Call ProcessOrder with 5 second deadline
            response = stub.ProcessOrder(request, timeout=5)
            print(f"Request {i+1}: SUCCESS - {response.message}")
            success_count += 1
        except grpc.RpcError as e:
            # Check if it's a timeout error
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print(f"Request {i+1}: TIMEOUT")
                timeout_count += 1
            else:
                print(f"Request {i+1}: ERROR - {e.code()}: {e.details()}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Results: {success_count} successful, {timeout_count} timeouts")
    print(f"{'='*50}")
    
    channel.close()


if __name__ == '__main__':
    run_test()