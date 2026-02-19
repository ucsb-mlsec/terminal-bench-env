/*
 * env_validator.c - SLURM SPANK Plugin for Environment Validation
 * 
 * This plugin validates environment variables for SLURM jobs
 * and ensures that required paths and configurations are properly set.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

const char plugin_name[] = "env_validator";
const char plugin_version[] = "2.1";

/* Environment variables that must be set */
static const char *required_vars[] = {
    "HOME",
    "USER",
    "PATH",
    NULL
};

/* Restricted environment variable prefixes */
static const char *restricted_prefixes[] = {
    "SLURM_",
    "PMI_",
    NULL
};

/*
 * Validate that required environment variables are present
 */
static int validate_required_env(spank_t sp)
{
    int i;
    char *value;
    
    for (i = 0; required_vars[i] != NULL; i++) {
        value = getenv(required_vars[i]);
        if (value == NULL || strlen(value) == 0) {
            fprintf(stderr, "env_validator: ERROR - Required environment variable %s is not set\n",
                    required_vars[i]);
            return -1;
        }
    }
    
    return 0;
}

/*
 * Check for restricted environment variable modifications
 */
static int check_restricted_vars(spank_t sp)
{
    extern char **environ;
    int i, j;
    
    for (i = 0; environ[i] != NULL; i++) {
        for (j = 0; restricted_prefixes[j] != NULL; j++) {
            if (strncmp(environ[i], restricted_prefixes[j], 
                       strlen(restricted_prefixes[j])) == 0) {
                fprintf(stderr, "env_validator: WARNING - Restricted variable detected: %s\n",
                        environ[i]);
            }
        }
    }
    
    return 0;
}

/*
 * Called during plugin initialization
 */
int slurm_spank_init(spank_t sp, int ac, char **av)
{
    int i;
    
    fprintf(stdout, "env_validator: Plugin version %s initializing\n", plugin_version);
    
    for (i = 0; i < ac; i++) {
        fprintf(stdout, "env_validator: Plugin argument %d: %s\n", i, av[i]);
    }
    
    return 0;
}

/*
 * Called after task fork in each task
 */
int slurm_spank_task_post_fork(spank_t sp, int ac, char **av)
{
    fprintf(stdout, "env_validator: Validating environment for task\n");
    
    if (validate_required_env(sp) != 0) {
        return -1;
    }
    
    check_restricted_vars(sp);
    
    fprintf(stdout, "env_validator: Environment validation completed successfully\n");
    
    return 0;
}