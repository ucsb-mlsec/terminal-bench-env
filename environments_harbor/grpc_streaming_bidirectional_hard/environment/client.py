#!/usr/bin/env python3

import grpc
import chat_pb2
import chat_pb2_grpc
import sys
import time
import threading
import random

def send_messages(username, num_messages):
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = chat_pb2_grpc.ChatServiceStub(channel)
        
        # BUG: Sending all messages first without proper generator pattern
        messages = []
        for i in range(num_messages):
            msg = chat_pb2.ChatMessage(
                username=username,
                message=f"Message {i} from {username}",
                timestamp=int(time.time())
            )
            messages.append(msg)
        
        # BUG: Not using proper bidirectional streaming - trying to send all then receive
        responses = stub.Chat(iter(messages))
        
        # BUG: Trying to read responses after sending all (blocking indefinitely)
        for response in responses:
            print(f"{response.username}: {response.message}")
            
    except Exception as e:
        print(f"Error: {e}")
        # BUG: Channel not properly closed

def receive_messages(stream):
    # BUG: Infinite blocking without timeout
    try:
        for response in stream:
            print(f"Received: {response.username}: {response.message}")
    except Exception as e:
        print(f"Receive error: {e}")

# BUG: Missing proper thread coordination between sending and receiving
# BUG: No reconnection logic when connection fails

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <username> [num_messages]")
        sys.exit(1)
    
    username = sys.argv[1]
    num_messages = 10
    
    if len(sys.argv) >= 3:
        num_messages = int(sys.argv[2])
    
    send_messages(username, num_messages)