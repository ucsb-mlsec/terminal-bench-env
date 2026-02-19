#include <slurm/spank.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

/*
 * SLURM SPANK Plugin: Resource Limiter
 * Enforces resource limits on jobs
 */

SPANK_PLUGIN(resource_limiter, 1);

const char plugin_name[] = "resource_limiter";
const char plugin_version[] = "1.0";

/* Maximum memory per task in MB */
#define MAX_MEMORY_MB 16384

/* Maximum CPU time per task in seconds */
#define MAX_CPU_TIME 86400

/*
 * Called from both srun and slurmd
 */
int slurm_spank_init(spank_t sp, int ac, char **av)
{
    int i;
    slurm_info("resource_limiter: plugin initializing");
    
    for (i = 0; i < ac; i++) {
        slurm_info("resource_limiter: arg %d: %s", i, av[i]);
    }
    
    return ESPANK_SUCCESS;
}

/*
 * Called in local context before job starts
 */
int slurm_spank_task_init(spank_t sp, int ac, char **av)
{
    struct rlimit mem_limit;
    struct rlimit cpu_limit;
    uint32_t job_id;
    uint32_t task_id;
    
    /* Get job and task information */
    if (spank_get_item(sp, S_JOB_ID, &job_id) != ESPANK_SUCCESS) {
        slurm_error("resource_limiter: failed to get job ID");
        return ESPANK_ERROR;
    }
    
    if (spank_get_item(sp, S_TASK_ID, &task_id) != ESPANK_SUCCESS) {
        slurm_error("resource_limiter: failed to get task ID");
        return ESPANK_ERROR;
    }
    
    slurm_info("resource_limiter: enforcing limits for job %u, task %u", 
               job_id, task_id);
    
    /* Set memory limit */
    mem_limit.rlim_cur = (rlim_t)MAX_MEMORY_MB * 1024 * 1024;
    mem_limit.rlim_max = (rlim_t)MAX_MEMORY_MB * 1024 * 1024;
    
    if (setrlimit(RLIMIT_AS, &mem_limit) != 0) {
        slurm_error("resource_limiter: failed to set memory limit");
        return ESPANK_ERROR;
    }
    
    /* Set CPU time limit */
    cpu_limit.rlim_cur = MAX_CPU_TIME;
    cpu_limit.rlim_max = MAX_CPU_TIME;
    
    if (setrlimit(RLIMIT_CPU, &cpu_limit) != 0) {
        slurm_error("resource_limiter: failed to set CPU time limit");
        return ESPANK_ERROR;
    }
    
    slurm_info("resource_limiter: limits applied successfully");
    
    return ESPANK_SUCCESS;
}

/*
 * Called before unloading the plugin
 */
int slurm_spank_exit(spank_t sp, int ac, char **av)
{
    slurm_info("resource_limiter: plugin exiting");
    return ESPANK_SUCCESS;
}