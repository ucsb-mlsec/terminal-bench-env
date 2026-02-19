#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sqlite3.h>

#define MAX_LINE_LENGTH 4096
#define MAX_COLUMNS 50

void print_usage(const char *prog_name) {
    printf("Usage: %s <command> [options]\n", prog_name);
    printf("\nCommands:\n");
    printf("  import <database> <csv_file> <table_name>\n");
    printf("      Import CSV file into SQLite database table\n");
    printf("  query <database> <sql_query>\n");
    printf("      Execute a SQL query on the database\n");
    printf("  list <database>\n");
    printf("      List all tables in the database\n");
    printf("\nExample:\n");
    printf("  %s import mydata.db data.csv employees\n", prog_name);
    printf("  %s query mydata.db \"SELECT * FROM employees\"\n", prog_name);
}

int count_columns(const char *line) {
    int count = 1;
    int in_quotes = 0;
    
    for (const char *p = line; *p; p++) {
        if (*p == '"') {
            in_quotes = !in_quotes;
        } else if (*p == ',' && !in_quotes) {
            count++;
        }
    }
    return count;
}

void parse_csv_line(const char *line, char **fields, int *field_count) {
    int field_idx = 0;
    int in_quotes = 0;
    char buffer[MAX_LINE_LENGTH];
    int buf_idx = 0;
    
    for (const char *p = line; *p; p++) {
        if (*p == '"') {
            in_quotes = !in_quotes;
        } else if (*p == ',' && !in_quotes) {
            buffer[buf_idx] = '\0';
            fields[field_idx] = strdup(buffer);
            field_idx++;
            buf_idx = 0;
        } else if (*p != '\r' && *p != '\n') {
            buffer[buf_idx++] = *p;
        }
    }
    
    buffer[buf_idx] = '\0';
    fields[field_idx] = strdup(buffer);
    *field_count = field_idx + 1;
}

void free_fields(char **fields, int count) {
    for (int i = 0; i < count; i++) {
        if (fields[i]) {
            free(fields[i]);
            fields[i] = NULL;
        }
    }
}

int create_table_from_csv(sqlite3 *db, const char *table_name, char **headers, int column_count) {
    char sql[MAX_LINE_LENGTH];
    char *err_msg = NULL;
    int offset = 0;
    
    offset += snprintf(sql + offset, sizeof(sql) - offset, "CREATE TABLE IF NOT EXISTS %s (", table_name);
    
    for (int i = 0; i < column_count; i++) {
        offset += snprintf(sql + offset, sizeof(sql) - offset, "%s TEXT", headers[i]);
        if (i < column_count - 1) {
            offset += snprintf(sql + offset, sizeof(sql) - offset, ", ");
        }
    }
    
    snprintf(sql + offset, sizeof(sql) - offset, ");");
    
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        return 1;
    }
    
    return 0;
}

int insert_csv_row(sqlite3 *db, const char *table_name, char **fields, int field_count) {
    char sql[MAX_LINE_LENGTH];
    char placeholders[512];
    int offset = 0;
    
    for (int i = 0; i < field_count; i++) {
        offset += snprintf(placeholders + offset, sizeof(placeholders) - offset, "?");
        if (i < field_count - 1) {
            offset += snprintf(placeholders + offset, sizeof(placeholders) - offset, ", ");
        }
    }
    
    snprintf(sql, sizeof(sql), "INSERT INTO %s VALUES (%s);", table_name, placeholders);
    
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return 1;
    }
    
    for (int i = 0; i < field_count; i++) {
        sqlite3_bind_text(stmt, i + 1, fields[i], -1, SQLITE_TRANSIENT);
    }
    
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "Execution failed: %s\n", sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        return 1;
    }
    
    sqlite3_finalize(stmt);
    return 0;
}

int import_csv(const char *db_name, const char *csv_file, const char *table_name) {
    sqlite3 *db;
    FILE *fp;
    char line[MAX_LINE_LENGTH];
    char *headers[MAX_COLUMNS];
    char *fields[MAX_COLUMNS];
    int column_count = 0;
    int row_count = 0;
    
    int rc = sqlite3_open(db_name, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }
    
    fp = fopen(csv_file, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open CSV file: %s\n", csv_file);
        sqlite3_close(db);
        return 1;
    }
    
    if (fgets(line, sizeof(line), fp)) {
        parse_csv_line(line, headers, &column_count);
        
        if (create_table_from_csv(db, table_name, headers, column_count) != 0) {
            free_fields(headers, column_count);
            fclose(fp);
            sqlite3_close(db);
            return 1;
        }
    }
    
    sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
    
    while (fgets(line, sizeof(line), fp)) {
        int field_count = 0;
        parse_csv_line(line, fields, &field_count);
        
        if (field_count == column_count) {
            if (insert_csv_row(db, table_name, fields, field_count) == 0) {
                row_count++;
            }
        }
        
        free_fields(fields, field_count);
    }
    
    sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
    
    printf("Successfully imported %d rows into table '%s'\n", row_count, table_name);
    
    free_fields(headers, column_count);
    fclose(fp);
    sqlite3_close(db);
    
    return 0;
}

int execute_query(const char *db_name, const char *query) {
    sqlite3 *db;
    sqlite3_stmt *stmt;
    
    int rc = sqlite3_open(db_name, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }
    
    rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare query: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }
    
    int col_count = sqlite3_column_count(stmt);
    
    for (int i = 0; i < col_count; i++) {
        printf("%-20s", sqlite3_column_name(stmt, i));
    }
    printf("\n");
    
    for (int i = 0; i < col_count * 20; i++) {
        printf("-");
    }
    printf("\n");
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        for (int i = 0; i < col_count; i++) {
            const char *text = (const char *)sqlite3_column_text(stmt, i);
            printf("%-20s", text ? text : "NULL");
        }
        printf("\n");
    }
    
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    
    return 0;
}

int list_tables(const char *db_name) {
    const char *query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;";
    return execute_query(db_name, query);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *command = argv[1];
    
    if (strcmp(command, "import") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Error: import requires 3 arguments\n");
            print_usage(argv[0]);
            return 1;
        }
        return import_csv(argv[2], argv[3], argv[4]);
    } else if (strcmp(command, "query") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Error: query requires 2 arguments\n");
            print_usage(argv[0]);
            return 1;
        }
        return execute_query(argv[2], argv[3]);
    } else if (strcmp(command, "list") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Error: list requires 1 argument\n");
            print_usage(argv[0]);
            return 1;
        }
        return list_tables(argv[2]);
    } else {
        fprintf(stderr, "Error: Unknown command '%s'\n", command);
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}