#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>

static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
    int i;
    for(i = 0; i < argc; i++) {
        printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
    printf("\n");
    return 0;
}

int main(int argc, char* argv[]) {
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;
    const char *sql;

    printf("Starting database utility...\n");

    // Open database
    rc = sqlite3_open("test.db", &db);
    if(rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return 1;
    } else {
        printf("Opened database successfully\n");
    }

    // Create table
    sql = "CREATE TABLE IF NOT EXISTS users("  \
          "id INTEGER PRIMARY KEY AUTOINCREMENT," \
          "name TEXT NOT NULL);";

    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
    if(rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
        sqlite3_close(db);
        return 1;
    } else {
        printf("Table created successfully\n");
    }

    // Insert sample records
    sql = "INSERT INTO users (name) VALUES ('Alice');" \
          "INSERT INTO users (name) VALUES ('Bob');" \
          "INSERT INTO users (name) VALUES ('Charlie');";

    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
    if(rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
        sqlite3_close(db);
        return 1;
    } else {
        printf("Records inserted successfully\n");
    }

    // Query data
    sql = "SELECT * FROM users;";
    printf("\nQuerying database:\n");

    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
    if(rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
        sqlite3_close(db);
        return 1;
    } else {
        printf("Query executed successfully\n");
    }

    // Close database
    sqlite3_close(db);
    printf("\nDatabase operations completed successfully\n");

    return 0;
}