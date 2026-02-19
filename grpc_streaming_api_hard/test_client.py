#!/usr/bin/env python3

import grpc
import json
import sys
import time
import threading
import argparse
from concurrent import futures

import chat_pb2
import chat_pb2_grpc


class ChatClient:
    def __init__(self, username, scenario_file):
        self.username = username
        self.scenario_file = scenario_file
        self.messages_sent = 0
        self.messages_received = 0
        self.running = True
        self.channel = None
        self.stub = None
        
    def load_scenario(self):
        """Load test scenario from JSON file"""
        with open(self.scenario_file, 'r') as f:
            data = json.load(f)
            return data.get('messages', [])
    
    def generate_requests(self):
        """Generate message requests based on scenario"""
        messages = self.load_scenario()
        
        for msg_data in messages:
            if not self.running:
                break
                
            message = chat_pb2.ChatMessage(
                username=self.username,
                message=msg_data['message'],
                timestamp=int(time.time() * 1000)
            )
            
            yield message
            self.messages_sent += 1
            print(f"[{self.username}] Sent: {msg_data['message']}")
            
            # Wait for specified delay before next message
            delay = msg_data.get('delay', 0)
            if delay > 0:
                time.sleep(delay / 1000.0)  # Convert ms to seconds
    
    def receive_messages(self, response_iterator):
        """Receive messages from server in separate thread"""
        try:
            for response in response_iterator:
                if not self.running:
                    break
                    
                # Only count messages from other users
                if response.username != self.username:
                    self.messages_received += 1
                    print(f"[{self.username}] Received from {response.username}: {response.message}")
        except grpc.RpcError as e:
            if self.running:
                print(f"[{self.username}] Stream error: {e.code()}")
        except Exception as e:
            if self.running:
                print(f"[{self.username}] Error receiving messages: {e}")
    
    def run(self):
        """Main client execution"""
        try:
            # Connect to server
            self.channel = grpc.insecure_channel('localhost:50052')
            self.stub = chat_pb2_grpc.ChatServiceStub(self.channel)
            
            print(f"[{self.username}] Connected to chat server")
            
            # Start bidirectional streaming
            response_iterator = self.stub.ChatStream(self.generate_requests())
            
            # Start receive thread
            receive_thread = threading.Thread(
                target=self.receive_messages,
                args=(response_iterator,),
                daemon=True
            )
            receive_thread.start()
            
            # Wait for receive thread to complete
            receive_thread.join(timeout=10)
            
            # Small delay to ensure all messages are processed
            time.sleep(0.5)
            
        except grpc.RpcError as e:
            print(f"[{self.username}] RPC error: {e.code()} - {e.details()}")
        except Exception as e:
            print(f"[{self.username}] Error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown client"""
        self.running = False
        if self.channel:
            self.channel.close()
        
        print(f"\n[{self.username}] Summary:")
        print(f"  Messages sent: {self.messages_sent}")
        print(f"  Messages received: {self.messages_received}")
    
    def get_stats(self):
        """Return client statistics"""
        return {
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received
        }


def main():
    parser = argparse.ArgumentParser(description='Chat client for testing')
    parser.add_argument('--username', required=True, help='Client username')
    parser.add_argument('--scenario', required=True, help='Path to scenario JSON file')
    
    args = parser.parse_args()
    
    client = ChatClient(args.username, args.scenario)
    client.run()
    
    # Return stats for verification
    stats = client.get_stats()
    return stats


if __name__ == '__main__':
    main()
