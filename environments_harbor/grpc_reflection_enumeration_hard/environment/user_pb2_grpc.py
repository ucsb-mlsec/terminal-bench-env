#!/usr/bin/env python3

import grpc
import json
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc


def discover_grpc_services():
    """Discover gRPC services using reflection API."""
    result = {
        "reflection_enabled": False,
        "services": []
    }
    
    try:
        # Create an insecure channel to the gRPC server
        channel = grpc.insecure_channel('localhost:50051')
        
        # Create a stub for the reflection service
        stub = reflection_pb2_grpc.ServerReflectionStub(channel)
        
        # Create a request to list all services
        request = reflection_pb2.ServerReflectionRequest(
            list_services=""
        )
        
        # Make the reflection call
        responses = stub.ServerReflectionInfo(iter([request]))
        
        # Process the response
        for response in responses:
            if response.HasField('list_services_response'):
                result["reflection_enabled"] = True
                
                # Extract service names
                for service in response.list_services_response.service:
                    service_name = service.name
                    
                    # Exclude reflection services
                    if service_name not in [
                        'grpc.reflection.v1alpha.ServerReflection',
                        'grpc.reflection.v1.ServerReflection'
                    ]:
                        result["services"].append(service_name)
                
                break
        
        channel.close()
        
    except grpc.RpcError as e:
        # If reflection is not enabled, we'll get an error
        result["reflection_enabled"] = False
        result["services"] = []
    except Exception as e:
        # Any other error means reflection is not available
        result["reflection_enabled"] = False
        result["services"] = []
    
    return result


def main():
    # Discover services
    discovery_result = discover_grpc_services()
    
    # Save to JSON file
    with open('/tmp/grpc_discovery.json', 'w') as f:
        json.dump(discovery_result, f, indent=2)
    
    print(f"Discovery complete. Reflection enabled: {discovery_result['reflection_enabled']}")
    print(f"Services found: {discovery_result['services']}")


if __name__ == '__main__':
    main()