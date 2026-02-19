#!/usr/bin/env python3

import grpc
from concurrent import futures
import threading
import queue
import time
import sys

import chat_pb2
import chat_pb2_grpc


class ChatService(chat_pb2_grpc.ChatServiceServicer):
    def __init__(self):
        self.clients = []
        self.client_lock = threading.Lock()
    
    def Chat(self, request_iterator, context):
        # Create a queue for this client to receive messages
        client_queue = queue.Queue()
        
        # Add client queue to the list
        self.client_lock.acquire()
        self.clients.append(client_queue)
        self.client_lock.release()
        
        # Bug 1: Not releasing lock properly in error cases
        # Bug 2: Starting receiver thread before sender is ready
        
        def receive_messages():
            # Bug 3: Not handling exceptions in request_iterator
            for message in request_iterator:
                print(f"Received message from {message.username}: {message.content}")
                
                # Bug 4: Race condition - iterating without proper locking
                for client in self.clients:
                    # Bug 5: Putting message in own queue too
                    client.put(message)
                    
                # Bug 6: Not handling full queues
                time.sleep(0.01)
        
        # Bug 7: Starting thread but not joining or managing it properly
        receiver_thread = threading.Thread(target=receive_messages)
        receiver_thread.start()
        
        # Bug 8: Generator doesn't handle client disconnection cleanup
        try:
            while True:
                # Bug 9: Not checking if context is active
                message = client_queue.get(timeout=1.0)
                yield message
        except queue.Empty:
            # Bug 10: Exiting on timeout instead of continuing
            pass
        except Exception as e:
            print(f"Error in Chat stream: {e}")
        finally:
            # Bug 11: Lock acquisition without guaranteed release
            self.client_lock.acquire()
            if client_queue in self.clients:
                self.clients.remove(client_queue)
            # Bug 12: Not releasing lock in all cases
            self.client_lock.release()


def serve():
    # Bug 13: Small thread pool that could cause blocking
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("Chat server started on port 50052")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()