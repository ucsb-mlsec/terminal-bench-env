#!/usr/bin/env python3

import grpc
from concurrent import futures
import time
from grpc_reflection.v1alpha import reflection

# Import necessary protobuf modules
from google.protobuf import descriptor_pool
from google.protobuf import symbol_database
from google.protobuf.descriptor_pb2 import FileDescriptorProto, DescriptorProto, FieldDescriptorProto, MethodDescriptorProto, ServiceDescriptorProto
import google.protobuf.descriptor

# Manually create protobuf descriptors for our services

def create_calculator_descriptor():
    # Create Calculator service
    file_descriptor_proto = FileDescriptorProto()
    file_descriptor_proto.name = 'calculator.proto'
    file_descriptor_proto.package = 'calculator'
    
    # Add message: AddRequest
    add_request_msg = file_descriptor_proto.message_type.add()
    add_request_msg.name = 'AddRequest'
    field1 = add_request_msg.field.add()
    field1.name = 'a'
    field1.number = 1
    field1.type = FieldDescriptorProto.TYPE_INT32
    field1.label = FieldDescriptorProto.LABEL_OPTIONAL
    field2 = add_request_msg.field.add()
    field2.name = 'b'
    field2.number = 2
    field2.type = FieldDescriptorProto.TYPE_INT32
    field2.label = FieldDescriptorProto.LABEL_OPTIONAL
    
    # Add message: AddResponse
    add_response_msg = file_descriptor_proto.message_type.add()
    add_response_msg.name = 'AddResponse'
    field3 = add_response_msg.field.add()
    field3.name = 'result'
    field3.number = 1
    field3.type = FieldDescriptorProto.TYPE_INT32
    field3.label = FieldDescriptorProto.LABEL_OPTIONAL
    
    # Add service: Calculator
    service = file_descriptor_proto.service.add()
    service.name = 'Calculator'
    
    method1 = service.method.add()
    method1.name = 'Add'
    method1.input_type = '.calculator.AddRequest'
    method1.output_type = '.calculator.AddResponse'
    
    method2 = service.method.add()
    method2.name = 'Subtract'
    method2.input_type = '.calculator.AddRequest'
    method2.output_type = '.calculator.AddResponse'
    
    return file_descriptor_proto

def create_user_descriptor():
    file_descriptor_proto = FileDescriptorProto()
    file_descriptor_proto.name = 'user.proto'
    file_descriptor_proto.package = 'user'
    
    # Add message: UserRequest
    user_request_msg = file_descriptor_proto.message_type.add()
    user_request_msg.name = 'UserRequest'
    field1 = user_request_msg.field.add()
    field1.name = 'user_id'
    field1.number = 1
    field1.type = FieldDescriptorProto.TYPE_STRING
    field1.label = FieldDescriptorProto.LABEL_OPTIONAL
    
    # Add message: UserResponse
    user_response_msg = file_descriptor_proto.message_type.add()
    user_response_msg.name = 'UserResponse'
    field2 = user_response_msg.field.add()
    field2.name = 'username'
    field2.number = 1
    field2.type = FieldDescriptorProto.TYPE_STRING
    field2.label = FieldDescriptorProto.LABEL_OPTIONAL
    
    # Add service: UserService
    service = file_descriptor_proto.service.add()
    service.name = 'UserService'
    
    method1 = service.method.add()
    method1.name = 'GetUser'
    method1.input_type = '.user.UserRequest'
    method1.output_type = '.user.UserResponse'
    
    method2 = service.method.add()
    method2.name = 'CreateUser'
    method2.input_type = '.user.UserRequest'
    method2.output_type = '.user.UserResponse'
    
    return file_descriptor_proto

def create_inventory_descriptor():
    file_descriptor_proto = FileDescriptorProto()
    file_descriptor_proto.name = 'inventory.proto'
    file_descriptor_proto.package = 'inventory'
    
    # Add message: ItemRequest
    item_request_msg = file_descriptor_proto.message_type.add()
    item_request_msg.name = 'ItemRequest'
    field1 = item_request_msg.field.add()
    field1.name = 'item_id'
    field1.number = 1
    field1.type = FieldDescriptorProto.TYPE_STRING
    field1.label = FieldDescriptorProto.LABEL_OPTIONAL
    
    # Add message: ItemResponse
    item_response_msg = file_descriptor_proto.message_type.add()
    item_response_msg.name = 'ItemResponse'
    field2 = item_response_msg.field.add()
    field2.name = 'quantity'
    field2.number = 1
    field2.type = FieldDescriptorProto.TYPE_INT32
    field2.label = FieldDescriptorProto.LABEL_OPTIONAL
    
    # Add service: InventoryManager
    service = file_descriptor_proto.service.add()
    service.name = 'InventoryManager'
    
    method1 = service.method.add()
    method1.name = 'CheckStock'
    method1.input_type = '.inventory.ItemRequest'
    method1.output_type = '.inventory.ItemResponse'
    
    method2 = service.method.add()
    method2.name = 'UpdateStock'
    method2.input_type = '.inventory.ItemRequest'
    method2.output_type = '.inventory.ItemResponse'
    
    return file_descriptor_proto

# Create generic handlers
class CalculatorServicer:
    def Add(self, request, context):
        return b''
    
    def Subtract(self, request, context):
        return b''

class UserServicer:
    def GetUser(self, request, context):
        return b''
    
    def CreateUser(self, request, context):
        return b''

class InventoryServicer:
    def CheckStock(self, request, context):
        return b''
    
    def UpdateStock(self, request, context):
        return b''

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Build file descriptors
    pool = descriptor_pool.Default()
    
    calc_fd_proto = create_calculator_descriptor()
    calc_fd = pool.Add(calc_fd_proto)
    
    user_fd_proto = create_user_descriptor()
    user_fd = pool.Add(user_fd_proto)
    
    inv_fd_proto = create_inventory_descriptor()
    inv_fd = pool.Add(inv_fd_proto)
    
    # Create generic RPC handlers for each service
    calculator_handler = grpc.method_handlers_generic_handler(
        'calculator.Calculator',
        {
            'Add': grpc.unary_unary_rpc_method_handler(
                CalculatorServicer().Add,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
            'Subtract': grpc.unary_unary_rpc_method_handler(
                CalculatorServicer().Subtract,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
        }
    )
    
    user_handler = grpc.method_handlers_generic_handler(
        'user.UserService',
        {
            'GetUser': grpc.unary_unary_rpc_method_handler(
                UserServicer().GetUser,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
            'CreateUser': grpc.unary_unary_rpc_method_handler(
                UserServicer().CreateUser,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
        }
    )
    
    inventory_handler = grpc.method_handlers_generic_handler(
        'inventory.InventoryManager',
        {
            'CheckStock': grpc.unary_unary_rpc_method_handler(
                InventoryServicer().CheckStock,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
            'UpdateStock': grpc.unary_unary_rpc_method_handler(
                InventoryServicer().UpdateStock,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            ),
        }
    )
    
    server.add_generic_rpc_handlers((calculator_handler,))
    server.add_generic_rpc_handlers((user_handler,))
    server.add_generic_rpc_handlers((inventory_handler,))
    
    # Enable reflection
    SERVICE_NAMES = (
        'calculator.Calculator',
        'user.UserService',
        'inventory.InventoryManager',
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()