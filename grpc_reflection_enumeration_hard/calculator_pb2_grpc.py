#!/usr/bin/env python3

import json
import grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc


def discover_grpc_services():
    """Discover gRPC services using reflection API."""
    try:
        # Connect to the gRPC server
        channel = grpc.insecure_channel('localhost:50051')
        
        # Create a reflection stub
        stub = reflection_pb2_grpc.ServerReflectionStub(channel)
        
        # Create a request to list services
        request = reflection_pb2.ServerReflectionRequest(
            list_services=""
        )
        
        # Send the request
        responses = stub.ServerReflectionInfo(iter([request]))
        
        # Parse the response
        services = []
        for response in responses:
            if response.HasField('list_services_response'):
                for service in response.list_services_response.service:
                    service_name = service.name
                    # Exclude reflection services
                    if not service_name.startswith('grpc.reflection.'):
                        services.append(service_name)
                break
        
        # Prepare result
        result = {
            "reflection_enabled": True,
            "services": sorted(services)
        }
        
    except grpc.RpcError as e:
        # Reflection is not enabled or service is not accessible
        result = {
            "reflection_enabled": False,
            "services": []
        }
    except Exception as e:
        # Any other error means reflection is not working
        result = {
            "reflection_enabled": False,
            "services": []
        }
    
    # Write result to file
    with open('/tmp/grpc_discovery.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


if __name__ == "__main__":
    result = discover_grpc_services()
    print(json.dumps(result, indent=2))