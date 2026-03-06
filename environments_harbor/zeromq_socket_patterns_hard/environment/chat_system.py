#!/usr/bin/env python3

import zmq
import threading
import time
import json
import sys
import argparse
import queue
import random
from collections import defaultdict
from datetime import datetime

class MessageBroker:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind("tcp://*:5555")
        self.clients = {}  # FIXME: not thread-safe
        self.running = True
        # TODO: add message acknowledgment system
        # TODO: add message queuing for offline clients
        
    def run(self):
        print("Message broker started on port 5555")
        while self.running:
            try:
                # FIXME: no timeout, blocks forever
                identity, empty, message = self.socket.recv_multipart()
                # TODO: handle disconnections properly - crashes here
                self.clients[identity] = time.time()
                
                # Simple broadcast - FIXME: loses messages during high load
                for client_id in self.clients.keys():
                    if client_id != identity:
                        try:
                            # FIXME: no error handling, no acknowledgment
                            self.socket.send_multipart([client_id, b'', message])
                        except:
                            pass  # TODO: handle send failures
            except Exception as e:
                # TODO: proper error handling
                print(f"Error: {e}")
                pass
    
    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()

class PresenceService:
    def __init__(self):
        self.client_status = {}  # FIXME: race conditions here
        # TODO: add proper synchronization
        
    def update_status(self, client_id, status):
        # FIXME: no locking, race conditions
        self.client_status[client_id] = {
            'status': status,
            'timestamp': time.time()
        }
    
    def get_status(self, client_id):
        # TODO: handle missing clients
        return self.client_status.get(client_id, {}).get('status', 'offline')
    
    def get_online_clients(self):
        # FIXME: doesn't check timestamp, stale data
        return [cid for cid, info in self.client_status.items() 
                if info.get('status') == 'online']

class ChatClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, client_id)
        # TODO: add sequence numbers for ordering
        # TODO: add message deduplication
        self.received_messages = []
        self.running = True
        
    def connect(self):
        # FIXME: unreliable reconnection logic
        self.socket.connect("tcp://localhost:5555")
        print(f"Client {self.client_id} connected")
    
    def send_message(self, content):
        # TODO: add acknowledgment handling
        # FIXME: no sequence numbers
        message = json.dumps({
            'from': self.client_id,
            'content': content,
            'timestamp': time.time()
        })
        self.socket.send_string(message)
    
    def receive_messages(self):
        while self.running:
            try:
                # FIXME: no timeout, blocks
                message = self.socket.recv_string()
                # TODO: check for duplicates
                # TODO: verify ordering
                self.received_messages.append(message)
            except:
                # TODO: handle network interruptions
                pass
    
    def reconnect(self):
        # FIXME: doesn't sync message history
        # TODO: implement proper reconnection with history
        self.socket.close()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.client_id)
        self.connect()
    
    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()

class MessagePersistence:
    def __init__(self):
        self.messages = []  # FIXME: no proper ordering guarantees
        # TODO: add persistence to disk
        
    def store_message(self, message):
        # FIXME: no locking, race conditions
        self.messages.append({
            'message': message,
            'timestamp': time.time()
        })
    
    def get_messages(self, since=None):
        # TODO: implement filtering by timestamp
        return self.messages
    
    def get_message_count(self):
        return len(self.messages)

def run_broker():
    broker = MessageBroker()
    presence = PresenceService()
    persistence = MessagePersistence()
    
    broker_thread = threading.Thread(target=broker.run)
    broker_thread.start()
    
    try:
        broker_thread.join()
    except KeyboardInterrupt:
        broker.stop()

def run_client(client_id):
    client = ChatClient(client_id)
    client.connect()
    
    receive_thread = threading.Thread(target=client.receive_messages)
    receive_thread.start()
    
    try:
        while True:
            msg = input(f"{client_id}> ")
            client.send_message(msg)
    except KeyboardInterrupt:
        client.stop()

def run_tests():
    print("Running test suite...")
    
    # Start broker
    broker = MessageBroker()
    broker_thread = threading.Thread(target=broker.run)
    broker_thread.daemon = True
    broker_thread.start()
    time.sleep(1)
    
    # Test 1: Basic message delivery
    print("Test 1: Basic message delivery")
    clients = []
    for i in range(5):
        client = ChatClient(f"client_{i}")
        client.connect()
        recv_thread = threading.Thread(target=client.receive_messages)
        recv_thread.daemon = True
        recv_thread.start()
        clients.append(client)
    
    time.sleep(1)
    
    # Send messages
    total_sent = 0
    for i, client in enumerate(clients):
        for j in range(20):
            client.send_message(f"Message {j} from client {i}")
            total_sent += 1
            time.sleep(0.01)  # FIXME: still loses messages
    
    time.sleep(2)
    
    # Check delivery
    total_received = sum(len(c.received_messages) for c in clients)
    expected = total_sent * (len(clients) - 1)  # Each message to all other clients
    delivery_rate = (total_received / expected * 100) if expected > 0 else 0
    
    print(f"Sent: {total_sent}, Expected receives: {expected}, Actual: {total_received}")
    print(f"Delivery rate: {delivery_rate:.2f}%")
    
    # Test 2: High load scenario (FIXME: fails here)
    print("\nTest 2: High load scenario")
    # TODO: implement concurrent sends
    
    # Test 3: Ordering (FIXME: not implemented)
    print("\nTest 3: Message ordering")
    ordering_violations = 0
    # TODO: check message sequence
    
    # Test 4: Disconnection handling (FIXME: crashes)
    print("\nTest 4: Client disconnection")
    # TODO: test abrupt disconnections
    
    # Cleanup
    for client in clients:
        client.stop()
    broker.stop()
    
    # Output results - FIXME: tests fail
    results = {
        "all_tests_passed": False,  # FIXME: should be true
        "message_delivery_rate": round(delivery_rate, 2),
        "ordering_violations": ordering_violations,
        "total_messages_tested": total_sent
    }
    
    with open('/home/user/solution/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to /home/user/solution/test_results.json")
    print(json.dumps(results, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Distributed Chat System')
    parser.add_argument('mode', choices=['broker', 'client', 'test'],
                       help='Mode to run: broker, client, or test')
    parser.add_argument('client_id', nargs='?', help='Client ID (for client mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'broker':
        run_broker()
    elif args.mode == 'client':
        if not args.client_id:
            print("Error: client_id required for client mode")
            sys.exit(1)
        run_client(args.client_id)
    elif args.mode == 'test':
        run_tests()

if __name__ == '__main__':
    main()