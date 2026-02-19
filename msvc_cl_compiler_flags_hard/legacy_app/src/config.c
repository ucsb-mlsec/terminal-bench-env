#include "../include/config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static Config global_config;
static int config_initialized = 0;

int init_config(void) {
    if (config_initialized) {
        return 0;
    }
    
    memset(&global_config, 0, sizeof(Config));
    load_defaults();
    config_initialized = 1;
    
    return 0;
}

int load_defaults(void) {
    global_config.debug_level = 1;
    global_config.operation_mode = MODE_NORMAL;
    global_config.timeout_value = 5000;
    global_config.max_retries = 3;
    global_config.buffer_size = 1024;
    global_config.enable_logging = 1;
    
    return 0;
}

int set_parameter(const char* param_name, int value) {
    if (!config_initialized) {
        return -1;
    }
    
    if (strcmp(param_name, "debug_level") == 0) {
        if (value >= 0 && value <= 5) {
            global_config.debug_level = value;
            return 0;
        }
    } else if (strcmp(param_name, "operation_mode") == 0) {
        if (value >= MODE_NORMAL && value <= MODE_ADVANCED) {
            global_config.operation_mode = value;
            return 0;
        }
    } else if (strcmp(param_name, "timeout_value") == 0) {
        if (value > 0 && value <= 60000) {
            global_config.timeout_value = value;
            return 0;
        }
    } else if (strcmp(param_name, "max_retries") == 0) {
        if (value >= 0 && value <= 10) {
            global_config.max_retries = value;
            return 0;
        }
    } else if (strcmp(param_name, "buffer_size") == 0) {
        if (value >= 256 && value <= 8192) {
            global_config.buffer_size = value;
            return 0;
        }
    } else if (strcmp(param_name, "enable_logging") == 0) {
        global_config.enable_logging = (value != 0);
        return 0;
    }
    
    return -1;
}

int get_parameter(const char* param_name) {
    if (!config_initialized) {
        return -1;
    }
    
    if (strcmp(param_name, "debug_level") == 0) {
        return global_config.debug_level;
    } else if (strcmp(param_name, "operation_mode") == 0) {
        return global_config.operation_mode;
    } else if (strcmp(param_name, "timeout_value") == 0) {
        return global_config.timeout_value;
    } else if (strcmp(param_name, "max_retries") == 0) {
        return global_config.max_retries;
    } else if (strcmp(param_name, "buffer_size") == 0) {
        return global_config.buffer_size;
    } else if (strcmp(param_name, "enable_logging") == 0) {
        return global_config.enable_logging;
    }
    
    return -1;
}

int validate_config(void) {
    if (!config_initialized) {
        return -1;
    }
    
    if (global_config.debug_level < 0 || global_config.debug_level > 5) {
        return -1;
    }
    
    if (global_config.operation_mode < MODE_NORMAL || 
        global_config.operation_mode > MODE_ADVANCED) {
        return -1;
    }
    
    if (global_config.timeout_value <= 0 || global_config.timeout_value > 60000) {
        return -1;
    }
    
    if (global_config.max_retries < 0 || global_config.max_retries > 10) {
        return -1;
    }
    
    if (global_config.buffer_size < 256 || global_config.buffer_size > 8192) {
        return -1;
    }
    
    return 0;
}

Config* get_config(void) {
    if (!config_initialized) {
        return NULL;
    }
    return &global_config;
}