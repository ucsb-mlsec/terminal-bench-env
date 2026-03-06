#ifndef DATABASE_H
#define DATABASE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NAME_LENGTH 100
#define MAX_RECORDS 1000

typedef struct {
    int id;
    char name[MAX_NAME_LENGTH];
    int age;
    double salary;
    int is_active;
} DatabaseRecord;

typedef struct {
    DatabaseRecord *records;
    int count;
    int capacity;
} DatabaseResult;

/* Function declarations for database operations */
int db_init(void);
int db_insert(const char *name, int age, double salary);
DatabaseResult* db_query(const char *name);
int db_update(int id, const char *name, int age, double salary);
int db_delete(int id);
void db_free_result(DatabaseResult *result);
void db_cleanup(void);

#endif