#!/usr/bin/env python3

import subprocess
import time
import threading
import os
import signal
import sys
import socket
from collections import defaultdict

# Global dictionary to track results
results = {
    'clients_connected': 0,
    'messages_sent': 0,
    'messages_received': 0,
    'message_loss': False,
    'order_preserved': True,
    'client_outputs': [],
    'lock': threading.Lock()
}

def run_client(client_id, num_messages=5):
    """
    Run a client subprocess and track its results
    """
    try:
        # Start client process
        process = subprocess.Popen(
            ['python3', 'client.py', f'user{client_id}', str(num_messages)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait with timeout
        try:
            stdout, stderr = process.communicate(timeout=30)
            
            with results['lock']:
                results['client_outputs'].append({
                    'client_id': client_id,
                    'stdout': stdout,
                    'stderr': stderr,
                    'returncode': process.returncode
                })
                
                # Update results based on output
                if 'Connected' in stdout or 'connected' in stdout.lower():
                    results['clients_connected'] += 1
                
                # Count messages sent and received
                for line in stdout.split('\n'):
                    if 'Sent' in line or 'sent' in line.lower():
                        results['messages_sent'] += 1
                    if 'Received' in line or 'received' in line.lower():
                        results['messages_received'] += 1
                    if 'out of order' in line.lower() or 'order' in line.lower():
                        results['order_preserved'] = False
                        
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            with results['lock']:
                results['client_outputs'].append({
                    'client_id': client_id,
                    'stdout': '',
                    'stderr': 'Timeout',
                    'returncode': -1
                })
                
    except Exception as e:
        with results['lock']:
            results['client_outputs'].append({
                'client_id': client_id,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            })

def is_server_running():
    """
    Check if server is still running by attempting connection
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 50051))
        sock.close()
        return result == 0
    except:
        return False

def main():
    """
    Main test function
    """
    print("Starting test scenario with 5 concurrent clients...")
    
    # Start 5 client threads concurrently
    threads = []
    for i in range(1, 6):
        thread = threading.Thread(target=run_client, args=(i, 5))
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Small delay between starts
    
    # Wait for all threads to complete with timeout
    for thread in threads:
        thread.join(timeout=35)
    
    # Give a moment for cleanup
    time.sleep(2)
    
    # Analyze results to determine message loss and order preservation
    expected_messages = 5 * 5  # 5 clients * 5 messages each
    
    if results['messages_received'] < results['messages_sent']:
        results['message_loss'] = True
    
    # Check if server is still running
    server_status = 'running' if is_server_running() else 'crashed'
    
    # Write results to file
    with open('/workspace/test_results.txt', 'w') as f:
        f.write(f"SERVER_STATUS={server_status}\n")
        f.write(f"CLIENTS_CONNECTED={results['clients_connected']}\n")
        f.write(f"MESSAGES_SENT={results['messages_sent']}\n")
        f.write(f"MESSAGES_RECEIVED={results['messages_received']}\n")
        f.write(f"MESSAGE_LOSS={'yes' if results['message_loss'] else 'no'}\n")
        f.write(f"ORDER_PRESERVED={'yes' if results['order_preserved'] else 'no'}\n")
    
    print(f"Test completed. Results written to /workspace/test_results.txt")
    print(f"Server Status: {server_status}")
    print(f"Clients Connected: {results['clients_connected']}")
    print(f"Messages Sent: {results['messages_sent']}")
    print(f"Messages Received: {results['messages_received']}")

if __name__ == '__main__':
    main()