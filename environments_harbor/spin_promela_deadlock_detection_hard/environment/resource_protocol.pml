/* Concurrent Resource Management System for Distributed Database */

byte resource1 = 0;  /* 0 = free, non-zero = locked by process ID */
byte resource2 = 0;
byte resource3 = 0;

/* Database Process Type 1 - Acquires resources in order: 1, 2, 3 */
proctype dbProcess1(byte pid) {
    /* Acquire resource1 */
    atomic {
        (resource1 == 0);
        resource1 = pid;
    }
    
    /* Simulate some processing delay */
    skip;
    
    /* Acquire resource2 */
    atomic {
        (resource2 == 0);
        resource2 = pid;
    }
    
    /* Acquire resource3 */
    atomic {
        (resource3 == 0);
        resource3 = pid;
    }
    
    /* Critical section - perform database operations */
    skip;
    
    /* Release all resources */
    resource1 = 0;
    resource2 = 0;
    resource3 = 0;
}

/* Database Process Type 2 - Acquires resources in order: 2, 3, 1 */
proctype dbProcess2(byte pid) {
    /* Acquire resource2 */
    atomic {
        (resource2 == 0);
        resource2 = pid;
    }
    
    skip;
    
    /* Acquire resource3 */
    atomic {
        (resource3 == 0);
        resource3 = pid;
    }
    
    /* Acquire resource1 */
    atomic {
        (resource1 == 0);
        resource1 = pid;
    }
    
    /* Critical section */
    skip;
    
    /* Release all resources */
    resource2 = 0;
    resource3 = 0;
    resource1 = 0;
}

/* Database Process Type 3 - Acquires resources in order: 3, 1, 2 */
proctype dbProcess3(byte pid) {
    /* Acquire resource3 */
    atomic {
        (resource3 == 0);
        resource3 = pid;
    }
    
    skip;
    
    /* Acquire resource1 */
    atomic {
        (resource1 == 0);
        resource1 = pid;
    }
    
    /* Acquire resource2 */
    atomic {
        (resource2 == 0);
        resource2 = pid;
    }
    
    /* Critical section */
    skip;
    
    /* Release all resources */
    resource3 = 0;
    resource1 = 0;
    resource2 = 0;
}

/* Initialize and start all processes */
init {
    run dbProcess1(1);
    run dbProcess2(2);
    run dbProcess3(3);
}