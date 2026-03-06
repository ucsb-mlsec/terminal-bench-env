#!/usr/bin/env python3

import grpc
import grpc.aio
import chat_pb2
import chat_pb2_grpc
import asyncio
import threading
import time

clients = []

class ChatServicer(chat_pb2_grpc.ChatServiceServicer):
    def StreamChat(self, request_iterator, context):
        clients.append(request_iterator)
        
        try:
            for message in request_iterator:
                print(f"Received message from {message.sender}: {message.content}")
                
                time.sleep(0.1)
                
                for client in clients:
                    try:
                        response = chat_pb2.ChatMessage(
                            sender=message.sender,
                            content=message.content,
                            timestamp=message.timestamp
                        )
                        yield response
                    except Exception as e:
                        print(f"Error sending to client: {e}")
                        
        except Exception as e:
            print(f"Error in StreamChat: {e}")
        
def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )
    chat_pb2_grpc.add_ChatServiceServicer_to_server(
        ChatServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()