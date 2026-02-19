#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>
#include "db_utils.h"

int create_test_table(sqlite3* db) {
    char *err_msg = NULL;
    const char *sql = "CREATE TABLE IF NOT EXISTS users ("
                     "id INTEGER PRIMARY KEY, "
                     "name TEXT);";
    
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err_msg);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to create table: %s\n", err_msg);
        sqlite3_free(err_msg);
        return 1;
    }
    
    printf("Table created successfully\n");
    return 0;
}