#!/usr/bin/env python3

import socket
import subprocess
import time
import threading
import sys

def test_calculation(calculation, expected, results, lock):
    """Send a calculation to the server and check the result."""
    try:
        # Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 5555))
        
        # Send calculation
        sock.sendall(calculation.encode('utf-8'))
        
        # Receive response
        response = sock.recv(1024).decode('utf-8').strip()
        
        # Close connection
        sock.close()
        
        # Check if response matches expected
        is_correct = (response == expected)
        
        with lock:
            results['total'] += 1
            if is_correct:
                results['correct'] += 1
                print(f"✓ {calculation} = {response} (expected {expected})")
            else:
                print(f"✗ {calculation} = {response} (expected {expected})")
                
    except Exception as e:
        with lock:
            results['total'] += 1
            print(f"✗ {calculation} failed with error: {e}")

def main():
    # Start the calculator server
    print("Starting calculator server...")
    server_process = subprocess.Popen(
        ['python3', '/home/agent/calculator_server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Define test cases: (calculation, expected_result)
    test_cases = [
        ('2+2', '4'),
        ('15+30', '45'),
        ('100+50', '150'),
        ('7+8', '15'),
        ('10-3', '7'),
        ('100-25', '75'),
        ('50-20', '30'),
        ('88-8', '80'),
        ('5*6', '30'),
        ('12*3', '36'),
        ('7*8', '56'),
        ('9*9', '81'),
        ('20/4', '5.0'),
        ('100/5', '20.0'),
        ('50/2', '25.0'),
        ('81/9', '9.0'),
        ('3+5*2', '13'),
        ('10-2*3', '4'),
        ('2++2', 'ERROR'),
        ('abc', 'ERROR'),
    ]
    
    # Results tracking
    results = {'correct': 0, 'total': 0}
    lock = threading.Lock()
    
    # Create threads for concurrent testing
    threads = []
    print(f"\nRunning {len(test_cases)} test calculations...")
    print("-" * 60)
    
    for calculation, expected in test_cases:
        thread = threading.Thread(
            target=test_calculation,
            args=(calculation, expected, results, lock)
        )
        threads.append(thread)
        thread.start()
        # Small delay to avoid overwhelming the server at once
        time.sleep(0.1)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("-" * 60)
    print(f"\nTest Results: {results['correct']}/{results['total']} correct")
    
    # Terminate the server
    print("Terminating server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        server_process.kill()
    
    # Write results to file
    with open('/home/agent/results.txt', 'w') as f:
        f.write(f"{results['correct']} {results['total']}\n")
    
    print(f"Results written to /home/agent/results.txt")
    
    # Exit with appropriate code
    sys.exit(0 if results['correct'] == results['total'] else 1)

if __name__ == '__main__':
    main()
```

Now, here's the calculator_server.py implementation:

```python
#!/usr/bin/env python3

import socket
import threading
import sys

def evaluate_expression(expression):
    """Safely evaluate a mathematical expression."""
    try:
        # Remove whitespace
        expression = expression.strip()
        
        # Check for empty expression
        if not expression:
            return "ERROR"
        
        # Only allow digits, operators, parentheses, and dots
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "ERROR"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return as string
        if isinstance(result, float) and result.is_integer():
            # Keep as float for division results
            if '/' in expression:
                return str(float(result))
            else:
                return str(int(result))
        else:
            return str(result)
            
    except Exception as e:
        return "ERROR"

def handle_client(client_socket, client_address):
    """Handle a single client connection."""
    try:
        # Receive calculation request
        data = client_socket.recv(1024).decode('utf-8')
        
        if data:
            # Evaluate the expression
            result = evaluate_expression(data)
            
            # Send result back to client
            client_socket.sendall(result.encode('utf-8'))
            
    except Exception as e:
        print(f"Error handling client {client_address}: {e}", file=sys.stderr)
    finally:
        # Close the client connection
        client_socket.close()

def main():
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Allow reuse of address
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind to port 5555
    server_socket.bind(('0.0.0.0', 5555))
    
    # Listen for connections
    server_socket.listen(5)
    
    print("Calculator server listening on port 5555...")
    
    try:
        while True:
            # Accept client connection
            client_socket, client_address = server_socket.accept()
            
            # Handle client in a new thread for concurrent processing
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address)
            )
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\nServer shutting down...")
    finally:
        server_socket.close()

if __name__ == '__main__':
    main()