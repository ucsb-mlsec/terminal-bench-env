#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "transaction.h"

// Transaction structure
struct Transaction {
    int id;
    double amount;
    char type[32];
    char description[128];
    int status;
};

// Transaction node for linked list
struct TransactionNode {
    Transaction* transaction;
    struct TransactionNode* next;
};

// Global transaction list head
static TransactionNode* transaction_list_head = NULL;

// Create a new transaction
Transaction* create_transaction(int id, double amount, const char* type, const char* description) {
    Transaction* trans = (Transaction*)malloc(sizeof(Transaction));
    if (!trans) {
        return NULL;
    }
    
    trans->id = id;
    trans->amount = amount;
    strncpy(trans->type, type, sizeof(trans->type) - 1);
    trans->type[sizeof(trans->type) - 1] = '\0';
    strncpy(trans->description, description, sizeof(trans->description) - 1);
    trans->description[sizeof(trans->description) - 1] = '\0';
    trans->status = 0; // 0 = pending
    
    return trans;
}

// Free a transaction
void free_transaction(Transaction* trans) {
    if (trans) {
        free(trans);
    }
}

// Copy transaction data
Transaction* copy_transaction(Transaction* src) {
    if (!src) {
        return NULL;
    }
    
    return create_transaction(src->id, src->amount, src->type, src->description);
}

// Get transaction ID
int get_transaction_id(Transaction* trans) {
    if (!trans) {
        return -1;
    }
    return trans->id;
}

// Get transaction amount
double get_transaction_amount(Transaction* trans) {
    if (!trans) {
        return 0.0;
    }
    return trans->amount;
}

// Set transaction status
void set_transaction_status(Transaction* trans, int status) {
    if (trans) {
        trans->status = status;
    }
}

// Get transaction status
int get_transaction_status(Transaction* trans) {
    if (!trans) {
        return -1;
    }
    return trans->status;
}

// USE-AFTER-FREE BUG #1: Process and finalize transaction
// This function frees the transaction but then accesses it
int process_transaction(Transaction* trans) {
    if (!trans) {
        return -1;
    }
    
    // Validate transaction amount
    if (trans->amount <= 0) {
        printf("Invalid transaction amount\n");
        return -1;
    }
    
    // Process the transaction
    printf("Processing transaction ID: %d\n", trans->id);
    printf("Amount: %.2f\n", trans->amount);
    
    // Free the transaction as it's been processed
    free_transaction(trans);
    
    // BUG: Accessing freed memory
    // Log the completion using the freed transaction
    printf("Completed transaction ID: %d with status: %d\n", trans->id, trans->status);
    
    return 0;
}

// Add transaction to list
int add_transaction_to_list(Transaction* trans) {
    if (!trans) {
        return -1;
    }
    
    TransactionNode* node = (TransactionNode*)malloc(sizeof(TransactionNode));
    if (!node) {
        return -1;
    }
    
    node->transaction = trans;
    node->next = transaction_list_head;
    transaction_list_head = node;
    
    return 0;
}

// USE-AFTER-FREE BUG #2: Remove transaction from list
// This function frees a node but then accesses it
int remove_transaction_from_list(int transaction_id) {
    TransactionNode* current = transaction_list_head;
    TransactionNode* previous = NULL;
    
    while (current != NULL) {
        if (current->transaction && current->transaction->id == transaction_id) {
            // Found the transaction to remove
            if (previous == NULL) {
                transaction_list_head = current->next;
            } else {
                previous->next = current->next;
            }
            
            // Free the transaction
            free_transaction(current->transaction);
            
            // Free the node
            free(current);
            
            // BUG: Accessing freed memory
            // Try to log information from the freed node
            printf("Removed transaction from list, next node: %p\n", (void*)current->next);
            
            return 0;
        }
        previous = current;
        current = current->next;
    }
    
    return -1; // Not found
}

// USE-AFTER-FREE BUG #3: Validate and return transaction
// This function frees the transaction on error but caller still uses the pointer
Transaction* validate_transaction(Transaction* trans) {
    if (!trans) {
        return NULL;
    }
    
    // Check transaction type
    if (strcmp(trans->type, "debit") != 0 && strcmp(trans->type, "credit") != 0) {
        printf("Invalid transaction type: %s\n", trans->type);
        // Free the invalid transaction
        free_transaction(trans);
        // BUG: Returning pointer to freed memory
        return trans;
    }
    
    // Check amount range
    if (trans->amount < 0.01 || trans->amount > 1000000.0) {
        printf("Transaction amount out of range: %.2f\n", trans->amount);
        // Free the invalid transaction
        free_transaction(trans);
        // BUG: Returning pointer to freed memory
        return trans;
    }
    
    // Transaction is valid
    trans->status = 1; // Valid
    return trans;
}

// Apply validation and use result
int apply_transaction_validation(Transaction* trans) {
    if (!trans) {
        return -1;
    }
    
    Transaction* validated = validate_transaction(trans);
    
    // BUG: Using potentially freed memory
    // If validation failed, validated points to freed memory
    if (validated && validated->status == 1) {
        printf("Transaction validated successfully\n");
        return 0;
    } else {
        // Still accessing the freed transaction
        printf("Transaction validation failed for ID: %d\n", validated->id);
        return -1;
    }
}

// Batch process transactions
int batch_process_transactions(Transaction** transactions, int count) {
    if (!transactions || count <= 0) {
        return -1;
    }
    
    for (int i = 0; i < count; i++) {
        if (transactions[i]) {
            process_transaction(transactions[i]);
        }
    }
    
    return 0;
}

// Clear all transactions from list
void clear_transaction_list() {
    TransactionNode* current = transaction_list_head;
    
    while (current != NULL) {
        TransactionNode* next = current->next;
        free_transaction(current->transaction);
        free(current);
        current = next;
    }
    
    transaction_list_head = NULL;
}

// Print transaction details
void print_transaction(Transaction* trans) {
    if (!trans) {
        printf("Transaction is NULL\n");
        return;
    }
    
    printf("Transaction Details:\n");
    printf("  ID: %d\n", trans->id);
    printf("  Amount: %.2f\n", trans->amount);
    printf("  Type: %s\n", trans->type);
    printf("  Description: %s\n", trans->description);
    printf("  Status: %d\n", trans->status);
}

// Count transactions in list
int count_transactions() {
    int count = 0;
    TransactionNode* current = transaction_list_head;
    
    while (current != NULL) {
        count++;
        current = current->next;
    }
    
    return count;
}

// Find transaction by ID
Transaction* find_transaction(int id) {
    TransactionNode* current = transaction_list_head;
    
    while (current != NULL) {
        if (current->transaction && current->transaction->id == id) {
            return current->transaction;
        }
        current = current->next;
    }
    
    return NULL;
}

// Update transaction amount
int update_transaction_amount(int id, double new_amount) {
    Transaction* trans = find_transaction(id);
    if (!trans) {
        return -1;
    }
    
    trans->amount = new_amount;
    return 0;
}

// Get total amount from all transactions
double get_total_amount() {
    double total = 0.0;
    TransactionNode* current = transaction_list_head;
    
    while (current != NULL) {
        if (current->transaction) {
            total += current->transaction->amount;
        }
        current = current->next;
    }
    
    return total;
}