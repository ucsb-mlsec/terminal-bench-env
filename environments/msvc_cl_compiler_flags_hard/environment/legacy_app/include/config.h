#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

/* Configuration parameter identifiers */
#define CONFIG_PARAM_TIMEOUT 0
#define CONFIG_PARAM_RETRY_COUNT 1
#define CONFIG_PARAM_BUFFER_SIZE 2
#define CONFIG_PARAM_DEBUG_LEVEL 3
#define CONFIG_PARAM_MAX_CONNECTIONS 4

/* Configuration value limits */
#define CONFIG_MAX_PARAMS 5
#define CONFIG_VALUE_MIN 0
#define CONFIG_VALUE_MAX 1000

/* Configuration structure */
struct config_data {
    int timeout;
    int retry_count;
    int buffer_size;
    int debug_level;
    int max_connections;
};

/* Function declarations */
int init_config(void);
int load_defaults(void);
int set_parameter(int param_id, int value);
int get_parameter(int param_id);
void cleanup_config(void);
struct config_data* get_config_data(void);

#endif /* CONFIG_H */