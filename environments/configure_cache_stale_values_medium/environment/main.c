#include <stdio.h>
#include <sqlite3.h>

int main() {
    sqlite3 *db;
    int rc;
    
    /* Open an in-memory database */
    rc = sqlite3_open(":memory:", &db);
    
    if (rc == SQLITE_OK) {
        printf("Successfully opened SQLite3 database\n");
        printf("SQLite version: %s\n", sqlite3_libversion());
        
        /* Close the database */
        sqlite3_close(db);
        
        return 0;
    } else {
        fprintf(stderr, "Failed to open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }
}