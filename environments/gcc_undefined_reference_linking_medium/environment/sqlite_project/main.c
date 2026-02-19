#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>
#include "db_utils.h"

int main() {
    sqlite3 *db;
    char *err_msg = NULL;
    int rc;
    
    printf("Starting database application...\n");
    
    // Open database
    rc = sqlite3_open("test.db", &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }
    
    printf("Database opened successfully\n");
    
    // Create test table using db_utils function
    if (create_test_table(db) != 0) {
        fprintf(stderr, "Failed to create table\n");
        sqlite3_close(db);
        return 1;
    }
    
    // Insert a test record
    const char *insert_sql = "INSERT INTO users (name, age) VALUES ('John Doe', 30);";
    rc = sqlite3_exec(db, insert_sql, NULL, NULL, &err_msg);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return 1;
    }
    
    printf("Test record inserted successfully\n");
    
    // Close database
    sqlite3_close(db);
    
    printf("Application completed successfully\n");
    
    return 0;
}