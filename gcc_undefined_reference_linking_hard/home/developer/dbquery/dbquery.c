#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>

int main(int argc, char *argv[]) {
    sqlite3 *db;
    sqlite3_stmt *stmt;
    int rc;
    
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <database_path> <sql_query>\n", argv[0]);
        return 1;
    }
    
    const char *db_path = argv[1];
    const char *sql_query = argv[2];
    
    // Open the database
    rc = sqlite3_open(db_path, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }
    
    // Prepare the SQL statement
    rc = sqlite3_prepare_v2(db, sql_query, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }
    
    // Execute the query and fetch results
    int row_count = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int col_count = sqlite3_column_count(stmt);
        
        for (int i = 0; i < col_count; i++) {
            int col_type = sqlite3_column_type(stmt, i);
            
            if (col_type == SQLITE_INTEGER) {
                int value = sqlite3_column_int(stmt, i);
                printf("%d", value);
            } else if (col_type == SQLITE_FLOAT) {
                double value = sqlite3_column_double(stmt, i);
                printf("%f", value);
            } else if (col_type == SQLITE_TEXT) {
                const unsigned char *value = sqlite3_column_text(stmt, i);
                printf("%s", value);
            } else if (col_type == SQLITE_NULL) {
                printf("NULL");
            }
            
            if (i < col_count - 1) {
                printf("|");
            }
        }
        printf("\n");
        row_count++;
    }
    
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "Error during query execution: %s\n", sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 1;
    }
    
    // Clean up
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    
    return 0;
}