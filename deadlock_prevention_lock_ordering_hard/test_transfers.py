#!/usr/bin/env python3

import threading
import time
import sys
from bank_system import BankSystem

def transfer_worker(bank, from_account, to_account, amount, thread_id):
    """Worker function to perform a transfer"""
    try:
        bank.transfer(from_account, to_account, amount)
        print(f"Thread {thread_id}: Transferred {amount} from {from_account} to {to_account}")
    except Exception as e:
        print(f"Thread {thread_id}: Transfer failed - {e}")

def main():
    # Create bank system
    bank = BankSystem()
    
    # Create accounts with initial balances
    accounts = ['A', 'B', 'C', 'D']
    initial_balance = 1000
    
    for account_id in accounts:
        bank.create_account(account_id, initial_balance)
    
    print("Initial balances:")
    for account_id in accounts:
        balance = bank.get_balance(account_id)
        print(f"  Account {account_id}: {balance}")
    
    # Calculate initial total
    initial_total = sum(bank.get_balance(acc) for acc in accounts)
    print(f"Initial total: {initial_total}\n")
    
    # Create threads for concurrent transfers
    threads = []
    thread_id = 0
    
    # These transfers will cause deadlock when they happen concurrently
    # A->B and B->A happening at the same time causes circular wait
    for i in range(10):
        t = threading.Thread(target=transfer_worker, args=(bank, 'A', 'B', 10, thread_id))
        threads.append(t)
        thread_id += 1
    
    for i in range(10):
        t = threading.Thread(target=transfer_worker, args=(bank, 'B', 'A', 15, thread_id))
        threads.append(t)
        thread_id += 1
    
    # C->D and D->C will also cause deadlock
    for i in range(10):
        t = threading.Thread(target=transfer_worker, args=(bank, 'C', 'D', 20, thread_id))
        threads.append(t)
        thread_id += 1
    
    for i in range(10):
        t = threading.Thread(target=transfer_worker, args=(bank, 'D', 'C', 25, thread_id))
        threads.append(t)
        thread_id += 1
    
    print(f"Starting {len(threads)} concurrent transfers...")
    print("This will likely deadlock due to opposite direction transfers...\n")
    
    # Start all threads
    start_time = time.time()
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    print(f"\nAll threads completed in {elapsed:.2f} seconds")
    
    # Print final balances
    print("\nFinal balances:")
    for account_id in accounts:
        balance = bank.get_balance(account_id)
        print(f"  Account {account_id}: {balance}")
    
    # Verify total balance
    final_total = sum(bank.get_balance(acc) for acc in accounts)
    print(f"Final total: {final_total}")
    
    if initial_total == final_total:
        print("\n✓ Balance verification passed!")
    else:
        print(f"\n✗ Balance verification FAILED! Lost/gained {final_total - initial_total}")
        sys.exit(1)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()