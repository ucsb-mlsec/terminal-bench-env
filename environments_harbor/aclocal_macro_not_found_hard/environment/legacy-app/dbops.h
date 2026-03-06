#ifndef DBOPS_H
#define DBOPS_H

/* Initialize database connection */
int db_init(const char *dbpath);

/* Log mathematical operation to database */
int db_log_operation(const char *operation, double result);

/* Close database connection */
void db_close(void);

#endif /* DBOPS_H */