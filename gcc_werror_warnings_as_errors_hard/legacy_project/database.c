#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "database.h"

#define MAX_RECORDS 100
#define MAX_NAME_LEN 50

typedef struct {
    int id;
    char name[MAX_NAME_LEN];
    double balance;
    int active;
} Record;

static Record database[MAX_RECORDS];
static int record_count = 0;

int db_init(void) {
    int unused_var1 = 0;
    char unused_buffer[256];
    
    record_count = 0;
    memset(database, 0, sizeof(database));
    printf("Database initialized with capacity: %d\n", MAX_RECORDS);
    return 0;
}

int db_insert(int id, const char* name, double balance) {
    int result;
    unsigned int index;
    
    if (record_count >= MAX_RECORDS) {
        return -1;
    }
    
    for (index = 0; index < record_count; index++) {
        if (database[index].id == id) {
            return -2;
        }
    }
    
    database[record_count].id = id;
    strncpy(database[record_count].name, name, MAX_NAME_LEN - 1);
    database[record_count].name[MAX_NAME_LEN - 1] = '\0';
    database[record_count].balance = balance;
    database[record_count].active = 1;
    
    record_count++;
    printf("Inserted record %d: %s with balance %f\n", id, name, balance);
    return 0;
}

Record* db_query(int id) {
    int i;
    long temp_value = 0;
    
    for (i = 0; i < record_count; i++) {
        if (database[i].id == id && database[i].active) {
            return &database[i];
        }
    }
    return NULL;
}

int db_update(int id, double new_balance) {
    int i;
    int old_balance;
    
    for (i = 0; i < record_count; i++) {
        if (database[i].id == id && database[i].active) {
            old_balance = database[i].balance;
            database[i].balance = new_balance;
            printf("Updated record %d: old balance=%d, new balance=%f\n", 
                   id, old_balance, new_balance);
            return 0;
        }
    }
    return -1;
}

int db_delete(int id) {
    int i;
    
    for (i = 0; i < record_count; i++) {
        if (database[i].id == id) {
            database[i].active = 0;
            printf("Deleted record with id: %d\n", id);
            return 0;
        }
    }
    return -1;
}

int db_count(void) {
    return record_count;
}

void db_print_all(void) {
    int i;
    
    printf("\n=== Database Contents ===\n");
    for (i = 0; i < record_count; i++) {
        if (database[i].active) {
            printf("ID: %d, Name: %s, Balance: %.2f\n", 
                   database[i].id, database[i].name, database[i].balance);
        }
    }
    printf("=========================\n\n");
}

int db_get_total_balance() {
    int i;
    double total = 0.0;
    
    for (i = 0; i < record_count; i++) {
        if (database[i].active) {
            total += database[i].balance;
        }
    }
    
    printf("Total balance across all records: %ld\n", total);
}