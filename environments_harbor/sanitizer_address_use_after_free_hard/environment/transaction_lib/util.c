#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "transaction.h"

/* Format a transaction amount as a currency string */
char* format_amount(double amount) {
    char* result = (char*)malloc(32);
    if (!result) return NULL;
    snprintf(result, 32, "$%.2f", amount);
    return result;
}

/* Format a timestamp as a readable date string */
char* format_date(time_t timestamp) {
    char* result = (char*)malloc(64);
    if (!result) return NULL;
    
    struct tm* timeinfo = localtime(&timestamp);
    strftime(result, 64, "%Y-%m-%d %H:%M:%S", timeinfo);
    return result;
}

/* Validate transaction amount */
int validate_amount(double amount) {
    if (amount < 0.0) {
        fprintf(stderr, "Error: Negative amount not allowed\n");
        return 0;
    }
    if (amount > 1000000.0) {
        fprintf(stderr, "Error: Amount exceeds maximum limit\n");
        return 0;
    }
    return 1;
}

/* Validate transaction ID */
int validate_transaction_id(const char* id) {
    if (!id) return 0;
    
    size_t len = strlen(id);
    if (len < 5 || len > 20) {
        fprintf(stderr, "Error: Invalid transaction ID length\n");
        return 0;
    }
    
    for (size_t i = 0; i < len; i++) {
        if (!((id[i] >= '0' && id[i] <= '9') || 
              (id[i] >= 'A' && id[i] <= 'Z') ||
              (id[i] >= 'a' && id[i] <= 'z') ||
              id[i] == '-' || id[i] == '_')) {
            fprintf(stderr, "Error: Invalid character in transaction ID\n");
            return 0;
        }
    }
    return 1;
}

/* Print transaction details for debugging */
void print_transaction(Transaction* trans) {
    if (!trans) {
        printf("Transaction: NULL\n");
        return;
    }
    
    printf("Transaction ID: %s\n", trans->id ? trans->id : "NULL");
    printf("Amount: %.2f\n", trans->amount);
    printf("Type: %d\n", trans->type);
    printf("Status: %d\n", trans->status);
    
    char* date_str = format_date(trans->timestamp);
    if (date_str) {
        printf("Timestamp: %s\n", date_str);
        free(date_str);
    }
    
    if (trans->description) {
        printf("Description: %s\n", trans->description);
    }
    printf("---\n");
}

/* Copy transaction ID string */
char* copy_transaction_id(const char* id) {
    if (!id) return NULL;
    
    size_t len = strlen(id);
    char* copy = (char*)malloc(len + 1);
    if (!copy) return NULL;
    
    strcpy(copy, id);
    return copy;
}

/* Copy transaction description */
char* copy_description(const char* desc) {
    if (!desc) return NULL;
    
    size_t len = strlen(desc);
    char* copy = (char*)malloc(len + 1);
    if (!copy) return NULL;
    
    strcpy(copy, desc);
    return copy;
}

/* Sanitize transaction description by removing special characters */
void sanitize_description(char* desc) {
    if (!desc) return;
    
    for (size_t i = 0; desc[i] != '\0'; i++) {
        if (desc[i] == '\n' || desc[i] == '\r' || desc[i] == '\t') {
            desc[i] = ' ';
        }
    }
}

/* Calculate transaction fee based on amount */
double calculate_fee(double amount) {
    if (amount < 0.0) return 0.0;
    
    if (amount <= 100.0) {
        return 1.0;
    } else if (amount <= 1000.0) {
        return amount * 0.02;
    } else {
        return amount * 0.015;
    }
}

/* Check if transaction is valid for processing */
int is_transaction_valid(Transaction* trans) {
    if (!trans) return 0;
    if (!trans->id) return 0;
    if (!validate_transaction_id(trans->id)) return 0;
    if (!validate_amount(trans->amount)) return 0;
    if (trans->type < 0 || trans->type > 3) return 0;
    return 1;
}

/* Log transaction error */
void log_error(const char* transaction_id, const char* error_msg) {
    fprintf(stderr, "[ERROR] Transaction %s: %s\n", 
            transaction_id ? transaction_id : "UNKNOWN", 
            error_msg ? error_msg : "Unknown error");
}

/* Log transaction info */
void log_info(const char* transaction_id, const char* info_msg) {
    printf("[INFO] Transaction %s: %s\n", 
           transaction_id ? transaction_id : "UNKNOWN", 
           info_msg ? info_msg : "");
}