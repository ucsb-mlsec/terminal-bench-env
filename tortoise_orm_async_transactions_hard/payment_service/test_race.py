#!/usr/bin/env python3

import threading
from . import db
from . import transfer

def main():
    # Set up database connection
    conn = db.get_connection()
    
    # Initialize the database
    db.init_db(conn)
    
    # Reset accounts to initial state
    db.reset_accounts(conn)
    
    # Print initial balances
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM accounts WHERE id = 1")
    balance1_initial = cursor.fetchone()[0]
    cursor.execute("SELECT balance FROM accounts WHERE id = 2")
    balance2_initial = cursor.fetchone()[0]
    
    print("Initial Balances:")
    print(f"  Account 1: ${balance1_initial:.2f}")
    print(f"  Account 2: ${balance2_initial:.2f}")
    print(f"  Total: ${balance1_initial + balance2_initial:.2f}")
    print()
    
    # Create 100 threads
    threads = []
    
    # 50 threads transferring $10 from account 1 to account 2
    for _ in range(50):
        thread = threading.Thread(target=transfer.transfer, args=(conn, 1, 2, 10))
        threads.append(thread)
    
    # 50 threads transferring $10 from account 2 to account 1
    for _ in range(50):
        thread = threading.Thread(target=transfer.transfer, args=(conn, 2, 1, 10))
        threads.append(thread)
    
    # Start all threads
    print("Starting 100 concurrent transfers...")
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All transfers completed.")
    print()
    
    # Query and print final balances
    cursor.execute("SELECT balance FROM accounts WHERE id = 1")
    balance1_final = cursor.fetchone()[0]
    cursor.execute("SELECT balance FROM accounts WHERE id = 2")
    balance2_final = cursor.fetchone()[0]
    
    print("Final Balances:")
    print(f"  Account 1: ${balance1_final:.2f}")
    print(f"  Account 2: ${balance2_final:.2f}")
    print(f"  Total: ${balance1_final + balance2_final:.2f}")
    print()
    
    # Determine if test passed or failed
    if balance1_final == 1000.00 and balance2_final == 1000.00:
        print("Result: PASS - Both accounts have exactly $1000.00")
    else:
        print(f"Result: FAIL - Expected both accounts to have $1000.00")
        print(f"         Account 1 has ${balance1_final:.2f} (off by ${balance1_final - 1000.00:.2f})")
        print(f"         Account 2 has ${balance2_final:.2f} (off by ${balance2_final - 1000.00:.2f})")
    
    conn.close()

if __name__ == "__main__":
    main()