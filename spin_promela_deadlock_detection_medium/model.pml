/* Promela Model: Resource Allocation with Potential Deadlock
   Two processes competing for two shared resources */

/* Shared resources represented as boolean flags */
bool resource1 = false;
bool resource2 = false;

/* Process A: Acquires resource1 first, then resource2 */
proctype ProcessA()
{
    printf("ProcessA: Starting\n");
    
    /* Try to acquire resource1 */
    atomic {
        !resource1;
        resource1 = true;
        printf("ProcessA: Acquired resource1\n");
    }
    
    /* Small delay to increase chance of interleaving */
    skip;
    
    /* Try to acquire resource2 */
    atomic {
        !resource2;
        resource2 = true;
        printf("ProcessA: Acquired resource2\n");
    }
    
    /* Critical section - both resources held */
    printf("ProcessA: In critical section\n");
    
    /* Release resources */
    resource2 = false;
    printf("ProcessA: Released resource2\n");
    resource1 = false;
    printf("ProcessA: Released resource1\n");
}

/* Process B: Acquires resource2 first, then resource1 (opposite order) */
proctype ProcessB()
{
    printf("ProcessB: Starting\n");
    
    /* Try to acquire resource2 */
    atomic {
        !resource2;
        resource2 = true;
        printf("ProcessB: Acquired resource2\n");
    }
    
    /* Small delay to increase chance of interleaving */
    skip;
    
    /* Try to acquire resource1 */
    atomic {
        !resource1;
        resource1 = true;
        printf("ProcessB: Acquired resource1\n");
    }
    
    /* Critical section - both resources held */
    printf("ProcessB: In critical section\n");
    
    /* Release resources */
    resource1 = false;
    printf("ProcessB: Released resource1\n");
    resource2 = false;
    printf("ProcessB: Released resource2\n");
}

/* Initialize and start both processes */
init
{
    atomic {
        run ProcessA();
        run ProcessB();
    }
}