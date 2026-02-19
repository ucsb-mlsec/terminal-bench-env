#!/usr/bin/env python3

import threading

class Account:
    def __init__(self, account_id, initial_balance):
        self.account_id = account_id
        self.balance = initial_balance
        self.lock = threading.Lock()
    
    def get_balance(self):
        return self.balance
    
    def deposit(self, amount):
        self.lock.acquire()
        self.balance += amount
        self.lock.release()
    
    def withdraw(self, amount):
        self.lock.acquire()
        if self.balance >= amount:
            self.balance -= amount
            self.lock.release()
            return True
        else:
            self.lock.release()
            return False


class BankSystem:
    def __init__(self):
        self.accounts = {}
    
    def create_account(self, account_id, initial_balance):
        if account_id not in self.accounts:
            self.accounts[account_id] = Account(account_id, initial_balance)
            return True
        return False
    
    def get_account(self, account_id):
        return self.accounts.get(account_id)
    
    def transfer(self, from_account_id, to_account_id, amount):
        from_account = self.get_account(from_account_id)
        to_account = self.get_account(to_account_id)
        
        if not from_account or not to_account:
            return False
        
        # DEADLOCK BUG: Acquiring locks in the order accounts are passed
        # This can cause circular waits when two threads transfer between
        # the same accounts in opposite directions
        from_account.lock.acquire()
        to_account.lock.acquire()
        
        if from_account.balance >= amount:
            from_account.balance -= amount
            to_account.balance += amount
            success = True
        else:
            success = False
        
        to_account.lock.release()
        from_account.lock.release()
        
        return success
    
    def get_total_balance(self):
        total = 0
        for account in self.accounts.values():
            total += account.get_balance()
        return total