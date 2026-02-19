#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "transaction.h"

void test_basic_transaction_creation() {
    printf("\n=== Test 1: Basic Transaction Creation ===\n");
    
    Transaction* t1 = create_transaction(1001, DEPOSIT, 500.0);
    if (t1) {
        printf("Created transaction ID: %d, Type: DEPOSIT, Amount: %.2f\n", 
               t1->id, t1->amount);
        process_transaction(t1);
        free_transaction(t1);
    }
    
    Transaction* t2 = create_transaction(1002, WITHDRAWAL, 200.0);
    if (t2) {
        printf("Created transaction ID: %d, Type: WITHDRAWAL, Amount: %.2f\n", 
               t2->id, t2->amount);
        process_transaction(t2);
        free_transaction(t2);
    }
}

void test_transfer_transactions() {
    printf("\n=== Test 2: Transfer Transactions ===\n");
    
    Transaction* t1 = create_transaction(2001, TRANSFER, 1000.0);
    if (t1) {
        printf("Created transfer transaction ID: %d, Amount: %.2f\n", 
               t1->id, t1->amount);
        process_transaction(t1);
        
        // Attempt to validate after processing
        if (validate_transaction(t1)) {
            printf("Transfer transaction validated successfully\n");
        }
        
        free_transaction(t1);
    }
    
    Transaction* t2 = create_transaction(2002, TRANSFER, 5000.0);
    if (t2) {
        printf("Created large transfer ID: %d, Amount: %.2f\n", 
               t2->id, t2->amount);
        process_transaction(t2);
        free_transaction(t2);
    }
}

void test_batch_processing() {
    printf("\n=== Test 3: Batch Transaction Processing ===\n");
    
    TransactionBatch* batch = create_batch(5);
    if (!batch) {
        printf("Failed to create batch\n");
        return;
    }
    
    printf("Created batch with capacity: %d\n", batch->capacity);
    
    Transaction* t1 = create_transaction(3001, DEPOSIT, 100.0);
    Transaction* t2 = create_transaction(3002, WITHDRAWAL, 50.0);
    Transaction* t3 = create_transaction(3003, TRANSFER, 200.0);
    Transaction* t4 = create_transaction(3004, DEPOSIT, 300.0);
    Transaction* t5 = create_transaction(3005, WITHDRAWAL, 150.0);
    
    add_to_batch(batch, t1);
    add_to_batch(batch, t2);
    add_to_batch(batch, t3);
    add_to_batch(batch, t4);
    add_to_batch(batch, t5);
    
    printf("Added %d transactions to batch\n", batch->count);
    
    process_batch(batch);
    
    printf("Batch processing completed\n");
    
    free_batch(batch);
}

void test_error_conditions() {
    printf("\n=== Test 4: Error Condition Handling ===\n");
    
    // Test with negative amount
    Transaction* t1 = create_transaction(4001, DEPOSIT, -100.0);
    if (t1) {
        printf("Created transaction with negative amount: %.2f\n", t1->amount);
        int result = process_transaction(t1);
        printf("Process result: %d\n", result);
        free_transaction(t1);
    }
    
    // Test with zero amount
    Transaction* t2 = create_transaction(4002, WITHDRAWAL, 0.0);
    if (t2) {
        printf("Created transaction with zero amount\n");
        process_transaction(t2);
        free_transaction(t2);
    }
    
    // Test cancellation
    Transaction* t3 = create_transaction(4003, TRANSFER, 500.0);
    if (t3) {
        printf("Created transaction for cancellation test\n");
        cancel_transaction(t3);
        printf("Transaction cancelled\n");
    }
}

void test_transaction_chain() {
    printf("\n=== Test 5: Transaction Chain Processing ===\n");
    
    Transaction* t1 = create_transaction(5001, DEPOSIT, 1000.0);
    Transaction* t2 = create_transaction(5002, WITHDRAWAL, 200.0);
    Transaction* t3 = create_transaction(5003, TRANSFER, 300.0);
    
    if (t1 && t2 && t3) {
        link_transactions(t1, t2);
        link_transactions(t2, t3);
        
        printf("Created transaction chain: %d -> %d -> %d\n", 
               t1->id, t2->id, t3->id);
        
        process_transaction_chain(t1);
        
        printf("Chain processing completed\n");
    }
}

void test_concurrent_operations() {
    printf("\n=== Test 6: Concurrent Operation Simulation ===\n");
    
    Transaction* t1 = create_transaction(6001, DEPOSIT, 500.0);
    if (t1) {
        printf("Transaction %d created\n", t1->id);
        
        // Simulate concurrent access
        validate_transaction(t1);
        process_transaction(t1);
        
        // Try to get transaction status
        int status = get_transaction_status(t1);
        printf("Transaction status: %d\n", status);
        
        free_transaction(t1);
    }
}

void test_rollback_scenario() {
    printf("\n=== Test 7: Transaction Rollback ===\n");
    
    Transaction* t1 = create_transaction(7001, TRANSFER, 2000.0);
    if (t1) {
        printf("Created transaction for rollback test\n");
        process_transaction(t1);
        
        // Attempt rollback
        rollback_transaction(t1);
        printf("Transaction rollback attempted\n");
    }
}

int main() {
    printf("Starting Transaction Library Tests\n");
    printf("===================================\n");
    
    test_basic_transaction_creation();
    test_transfer_transactions();
    test_batch_processing();
    test_error_conditions();
    test_transaction_chain();
    test_concurrent_operations();
    test_rollback_scenario();
    
    printf("\n===================================\n");
    printf("All tests completed\n");
    
    return 0;
}