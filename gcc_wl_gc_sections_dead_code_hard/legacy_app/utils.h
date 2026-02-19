#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

/* Used utility functions */
int calculate_checksum(const char *data, size_t len);
char* string_duplicate(const char *str);
void print_banner(const char *app_name);
int parse_config_value(const char *key);
void initialize_system(void);

/* Unused legacy functions */
void legacy_compute_hash(const char *input);
int deprecated_string_compare(const char *s1, const char *s2);
void old_formatting_routine(char *buffer, int size);
double unused_math_operation(double x, double y);
void debug_memory_dump(void *ptr, size_t size);
char* obsolete_string_parser(const char *input);
int legacy_error_handler(int error_code);
void unused_logging_function(const char *message);

/* Unused data structures */
struct legacy_config {
    int old_timeout;
    char old_path[256];
    int deprecated_flags;
};

/* Unused function pointers */
typedef int (*old_callback_t)(void *data);
typedef void (*unused_handler_t)(int signal);

/* More unused utilities */
void reset_unused_counters(void);
int validate_deprecated_format(const char *str);

#endif /* UTILS_H */