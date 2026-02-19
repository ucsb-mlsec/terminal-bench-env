#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "account.h"

#define NUM_ACCOUNTS 5
#define INITIAL_BALANCE 1000

typedef struct {
    int account_id;
    int balance;
    int transaction_count;
} Account;

Account accounts[NUM_ACCOUNTS];
int accounts_initialized = 0;

void init_accounts(void) {
    if (accounts_initialized) {
        return;
    }
    
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        accounts[i].account_id = i + 1;
        accounts[i].balance = INITIAL_BALANCE;
        accounts[i].transaction_count = 0;
    }
    
    accounts_initialized = 1;
    printf("Initialized %d accounts with balance %d each\n", NUM_ACCOUNTS, INITIAL_BALANCE);
}

int get_num_accounts(void) {
    return NUM_ACCOUNTS;
}

int get_balance(int account_id) {
    if (account_id < 1 || account_id > NUM_ACCOUNTS) {
        return -1;
    }
    
    int idx = account_id - 1;
    int balance = accounts[idx].balance;
    
    return balance;
}

int deposit(int account_id, int amount) {
    if (account_id < 1 || account_id > NUM_ACCOUNTS) {
        return -1;
    }
    
    if (amount <= 0) {
        return -1;
    }
    
    int idx = account_id - 1;
    
    // RACE CONDITION: Read-modify-write without synchronization
    int current_balance = accounts[idx].balance;
    
    // Simulate some processing time to increase likelihood of race
    for (volatile int i = 0; i < 100; i++);
    
    int new_balance = current_balance + amount;
    accounts[idx].balance = new_balance;
    
    // Update transaction count (also unsynchronized)
    int count = accounts[idx].transaction_count;
    accounts[idx].transaction_count = count + 1;
    
    return new_balance;
}

int withdraw(int account_id, int amount) {
    if (account_id < 1 || account_id > NUM_ACCOUNTS) {
        return -1;
    }
    
    if (amount <= 0) {
        return -1;
    }
    
    int idx = account_id - 1;
    
    // RACE CONDITION: Read-modify-write without synchronization
    int current_balance = accounts[idx].balance;
    
    if (current_balance < amount) {
        return -1; // Insufficient funds
    }
    
    // Simulate some processing time
    for (volatile int i = 0; i < 100; i++);
    
    int new_balance = current_balance - amount;
    accounts[idx].balance = new_balance;
    
    // Update transaction count (also unsynchronized)
    int count = accounts[idx].transaction_count;
    accounts[idx].transaction_count = count + 1;
    
    return new_balance;
}

int process_transaction(int account_id, int amount) {
    if (account_id < 1 || account_id > NUM_ACCOUNTS) {
        fprintf(stderr, "Invalid account ID: %d\n", account_id);
        return -1;
    }
    
    int result;
    
    if (amount > 0) {
        result = deposit(account_id, amount);
    } else if (amount < 0) {
        result = withdraw(account_id, -amount);
    } else {
        return get_balance(account_id);
    }
    
    return result;
}

int get_transaction_count(int account_id) {
    if (account_id < 1 || account_id > NUM_ACCOUNTS) {
        return -1;
    }
    
    int idx = account_id - 1;
    return accounts[idx].transaction_count;
}

void print_account_summary(void) {
    printf("\n===== Account Summary =====\n");
    int total_balance = 0;
    int total_transactions = 0;
    
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        printf("Account %d: Balance = %d, Transactions = %d\n",
               accounts[i].account_id,
               accounts[i].balance,
               accounts[i].transaction_count);
        total_balance += accounts[i].balance;
        total_transactions += accounts[i].transaction_count;
    }
    
    printf("\nTotal Balance: %d (Expected: %d)\n", 
           total_balance, NUM_ACCOUNTS * INITIAL_BALANCE);
    printf("Total Transactions: %d\n", total_transactions);
    printf("===========================\n\n");
}

int verify_balances(int expected_total) {
    int total = 0;
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        total += accounts[i].balance;
    }
    return (total == expected_total) ? 1 : 0;
}