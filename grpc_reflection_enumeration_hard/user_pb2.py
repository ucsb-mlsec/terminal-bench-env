#!/usr/bin/env python3

import grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc
import json

def check_grpc_reflection(host='localhost', port=50051):
    """Check if gRPC reflection is enabled and enumerate services."""
    
    channel = grpc.insecure_channel(f'{host}:{port}')
    
    try:
        # Create a stub for the reflection service
        stub = reflection_pb2_grpc.ServerReflectionStub(channel)
        
        # Create a request to list services
        request = reflection_pb2.ServerReflectionRequest(
            list_services=""
        )
        
        # Try to call the reflection service
        responses = stub.ServerReflectionInfo(iter([request]))
        
        services = []
        reflection_enabled = False
        
        for response in responses:
            if response.HasField('list_services_response'):
                reflection_enabled = True
                for service in response.list_services_response.service:
                    service_name = service.name
                    # Exclude the reflection services themselves
                    if service_name not in ['grpc.reflection.v1alpha.ServerReflection', 
                                           'grpc.reflection.v1.ServerReflection']:
                        services.append(service_name)
                break
        
        return {
            "reflection_enabled": reflection_enabled,
            "services": sorted(services)
        }
        
    except grpc.RpcError as e:
        # If we get an error, reflection is likely disabled
        return {
            "reflection_enabled": False,
            "services": []
        }
    except Exception as e:
        # Any other error means reflection is not available
        return {
            "reflection_enabled": False,
            "services": []
        }
    finally:
        channel.close()

def main():
    result = check_grpc_reflection()
    
    # Write results to file
    with open('/tmp/grpc_discovery.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Discovery complete. Results saved to /tmp/grpc_discovery.json")
    print(f"Reflection enabled: {result['reflection_enabled']}")
    if result['services']:
        print(f"Services found: {', '.join(result['services'])}")

if __name__ == '__main__':
    main()