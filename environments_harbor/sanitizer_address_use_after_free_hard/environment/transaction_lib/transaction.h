#ifndef TRANSACTION_H
#define TRANSACTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Transaction type definitions */
#define DEPOSIT 1
#define WITHDRAWAL 2
#define TRANSFER 3

/* Transaction status codes */
#define STATUS_PENDING 0
#define STATUS_COMPLETED 1
#define STATUS_FAILED 2

/* Structure representing a single transaction */
typedef struct Transaction {
    unsigned long transaction_id;
    double amount;
    time_t timestamp;
    int type;
    int status;
    char description[256];
    struct Transaction *next;
} Transaction;

/* Structure representing a list of transactions */
typedef struct TransactionList {
    Transaction *head;
    Transaction *tail;
    int count;
} TransactionList;

/* Structure for transaction processor */
typedef struct TransactionProcessor {
    TransactionList *pending_transactions;
    TransactionList *completed_transactions;
    double total_processed;
} TransactionProcessor;

/* Transaction creation and destruction */
Transaction* create_transaction(unsigned long id, double amount, int type, const char *description);
void free_transaction(Transaction *trans);

/* Transaction list operations */
TransactionList* create_transaction_list(void);
void add_transaction(TransactionList *list, Transaction *trans);
Transaction* remove_transaction(TransactionList *list, unsigned long id);
void free_transaction_list(TransactionList *list);

/* Transaction processor operations */
TransactionProcessor* create_processor(void);
void process_transaction(TransactionProcessor *processor, Transaction *trans);
void process_pending_transactions(TransactionProcessor *processor);
void free_processor(TransactionProcessor *processor);

/* Utility functions */
void print_transaction(Transaction *trans);
void print_transaction_list(TransactionList *list);
Transaction* find_transaction(TransactionList *list, unsigned long id);
int validate_transaction(Transaction *trans);

#endif /* TRANSACTION_H */