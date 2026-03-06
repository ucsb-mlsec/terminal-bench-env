/* Token Ring Protocol Model with Timing Constraints
 * Models a 3-process token ring where processes must wait
 * at least 2 time units between releasing and accepting a token.
 * Uses embedded C code to track timing state.
 */

#define NUM_PROCS 3
#define MIN_WAIT_TIME 2

/* Embedded C code to track timing state */
c_track "int last_release_time[3]" "int current_time"
c_track "int can_accept[3]"

c_code {
    int last_release_time[3] = {-10, -10, -10};  /* Initialize to allow immediate first accept */
    int current_time = 0;
    int can_accept[3] = {1, 1, 1};  /* All can initially accept */
    
    /* Update whether a process can accept based on timing */
    void update_can_accept(int pid) {
        if (current_time - last_release_time[pid] >= MIN_WAIT_TIME) {
            can_accept[pid] = 1;
        } else {
            can_accept[pid] = 0;
        }
    }
    
    /* Record that a process released the token */
    void record_release(int pid) {
        last_release_time[pid] = current_time;
        can_accept[pid] = 0;
    }
    
    /* Advance time */
    void advance_time() {
        current_time++;
        /* Update all processes after time advances */
        int i;
        for (i = 0; i < 3; i++) {
            update_can_accept(i);
        }
    }
}

/* Channel for token passing - buffered to allow asynchronous send/receive */
chan ring[NUM_PROCS] = [1] of { byte };

/* Track token ownership for verification */
byte token_holder = 0;  /* Process that currently holds token (0, 1, or 2) */
bool has_token[NUM_PROCS];  /* Array to track who has token */

/* Initialize: P0 starts with the token */
init {
    has_token[0] = true;
    has_token[1] = false;
    has_token[2] = false;
    
    /* Start all processes */
    run process(0);
    run process(1);
    run process(2);
    
    /* Start time advancement process */
    run time_keeper();
}

/* Process behavior */
proctype process(byte pid) {
    byte next_pid = (pid + 1) % NUM_PROCS;
    byte dummy;
    
    do
    :: has_token[pid] ->
        /* Process holds token, can pass it on */
        atomic {
            /* Pass token to next process */
            ring[next_pid] ! 1;
            has_token[pid] = false;
            
            /* Record release time in C code */
            c_code { record_release(Pprocess->pid); }
        }
    
    :: ring[pid] ? dummy ->
        /* Received token - check timing constraint */
        c_code { update_can_accept(Pprocess->pid); }
        
        c_expr { can_accept[Pprocess->pid] } ->
            atomic {
                has_token[pid] = true;
                
                /* Verify token count invariant */
                assert(has_token[0] + has_token[1] + has_token[2] == 1);
            }
    od
}

/* Time keeper process to advance simulation time */
proctype time_keeper() {
    do
    :: timeout ->
        c_code { advance_time(); }
    od
}

/* LTL properties for verification */

/* Progress property: eventually each process gets the token infinitely often */
ltl progress_p0 { []<> has_token[0] }
ltl progress_p1 { []<> has_token[1] }
ltl progress_p2 { []<> has_token[2] }

/* Token count invariant: exactly one token exists at all times */
ltl single_token { [] (has_token[0] + has_token[1] + has_token[2] == 1) }