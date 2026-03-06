#!/usr/bin/env python3

import grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc
import json

def check_grpc_reflection():
    """
    Check if gRPC reflection is enabled and enumerate services.
    """
    try:
        # Create a channel to the gRPC server
        channel = grpc.insecure_channel('localhost:50051')
        
        # Create a stub for the reflection service
        stub = reflection_pb2_grpc.ServerReflectionStub(channel)
        
        # Create a request to list services
        request = reflection_pb2.ServerReflectionRequest(
            list_services=""
        )
        
        # Make the reflection call
        responses = stub.ServerReflectionInfo(iter([request]))
        
        # Process responses
        services = []
        reflection_enabled = False
        
        for response in responses:
            if response.HasField('list_services_response'):
                reflection_enabled = True
                for service in response.list_services_response.service:
                    service_name = service.name
                    # Filter out the reflection service itself
                    if service_name not in ['grpc.reflection.v1alpha.ServerReflection', 
                                           'grpc.reflection.v1.ServerReflection']:
                        services.append(service_name)
                break
        
        # Prepare result
        result = {
            "reflection_enabled": reflection_enabled,
            "services": sorted(services)
        }
        
        # Write to file
        with open('/tmp/grpc_discovery.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Discovery complete. Reflection enabled: {reflection_enabled}")
        print(f"Services found: {services}")
        
    except grpc.RpcError as e:
        # Reflection is likely disabled
        result = {
            "reflection_enabled": False,
            "services": []
        }
        
        with open('/tmp/grpc_discovery.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Reflection appears to be disabled or service unreachable: {e}")
        
    except Exception as e:
        # Handle other errors
        result = {
            "reflection_enabled": False,
            "services": []
        }
        
        with open('/tmp/grpc_discovery.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Error during discovery: {e}")

if __name__ == "__main__":
    check_grpc_reflection()