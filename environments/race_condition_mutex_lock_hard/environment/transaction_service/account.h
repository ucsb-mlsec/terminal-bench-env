#ifndef ACCOUNT_H
#define ACCOUNT_H

#include <pthread.h>

#define NUM_ACCOUNTS 10
#define INITIAL_BALANCE 1000

typedef struct {
    int account_id;
    int balance;
    pthread_mutex_t lock;
} Account;

/**
 * Initialize all accounts with default values
 */
void init_accounts(void);

/**
 * Process a transaction from one account to another
 * Returns 0 on success, -1 on failure
 */
int process_transaction(int from_account, int to_account, int amount);

/**
 * Get the current balance of an account
 * Returns balance, or -1 on error
 */
int get_balance(int account_id);

/**
 * Deposit amount into an account
 * Returns 0 on success, -1 on failure
 */
int deposit(int account_id, int amount);

/**
 * Withdraw amount from an account
 * Returns 0 on success, -1 on failure
 */
int withdraw(int account_id, int amount);

/**
 * Get total balance across all accounts
 * Returns sum of all account balances
 */
int get_total_balance(void);

/**
 * Cleanup resources
 */
void cleanup_accounts(void);

#endif /* ACCOUNT_H */