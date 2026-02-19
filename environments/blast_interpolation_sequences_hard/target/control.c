#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define MAX_RESOURCES 100
#define MAX_BUFFER 50
#define STATE_IDLE 0
#define STATE_ALLOCATING 1
#define STATE_PROCESSING 2
#define STATE_RELEASING 3
#define MAX_ITERATIONS 20

/* Resource allocation control system
 * Manages resource allocation with buffer management
 * States: IDLE -> ALLOCATING -> PROCESSING -> RELEASING -> IDLE
 */

int main() {
    int resource_count = 0;      // Current allocated resources
    int buffer_level = 0;        // Buffer utilization level
    int state = STATE_IDLE;      // Current state machine state
    int pending_requests = 0;    // Pending allocation requests
    int iteration = 0;
    
    printf("Resource Allocation Control System\n");
    printf("Enter commands (1=allocate, 2=process, 3=release, 0=idle, -1=exit):\n");
    
    // Main control loop
    while (iteration < MAX_ITERATIONS) {
        int command;
        int amount;
        
        // Read command from input
        if (scanf("%d", &command) != 1) {
            break;
        }
        
        if (command == -1) {
            break;
        }
        
        // State machine transitions
        switch (state) {
            case STATE_IDLE:
                if (command == 1) {
                    // Transition to allocating state
                    if (scanf("%d", &amount) != 1) {
                        amount = 10;
                    }
                    pending_requests = amount;
                    state = STATE_ALLOCATING;
                    printf("IDLE -> ALLOCATING (requests: %d)\n", pending_requests);
                }
                break;
                
            case STATE_ALLOCATING:
                // Allocate resources based on pending requests
                if (command == 1) {
                    // Additional allocation request
                    if (scanf("%d", &amount) != 1) {
                        amount = 5;
                    }
                    pending_requests += amount;
                }
                
                // Process pending allocations
                if (pending_requests > 0) {
                    int allocate_amount = pending_requests;
                    if (allocate_amount > 15) {
                        allocate_amount = 15;  // Allocate in chunks
                    }
                    
                    resource_count += allocate_amount;
                    pending_requests -= allocate_amount;
                    buffer_level += allocate_amount / 2;  // Buffer grows with allocation
                    
                    printf("Allocated %d resources (total: %d, buffer: %d)\n", 
                           allocate_amount, resource_count, buffer_level);
                }
                
                // Check resource bounds
                assert(resource_count >= 0);
                assert(resource_count <= MAX_RESOURCES);
                
                if (command == 2 || pending_requests == 0) {
                    state = STATE_PROCESSING;
                    printf("ALLOCATING -> PROCESSING\n");
                }
                break;
                
            case STATE_PROCESSING:
                // Process resources and update buffer
                if (command == 2) {
                    if (scanf("%d", &amount) != 1) {
                        amount = 8;
                    }
                    
                    // Processing increases buffer usage
                    buffer_level += amount;
                    
                    // Some processing consumes resources
                    if (resource_count > 5) {
                        resource_count -= 3;
                    }
                    
                    printf("Processing: buffer_level=%d, resources=%d\n", 
                           buffer_level, resource_count);
                }
                
                // Critical buffer check
                assert(buffer_level < MAX_BUFFER);
                assert(buffer_level >= 0);
                
                if (command == 3) {
                    state = STATE_RELEASING;
                    printf("PROCESSING -> RELEASING\n");
                }
                
                // Auto-transition if buffer is getting full
                if (buffer_level > 35) {
                    state = STATE_RELEASING;
                    printf("Auto-transition PROCESSING -> RELEASING (buffer high)\n");
                }
                break;
                
            case STATE_RELEASING:
                // Release resources and clear buffer
                if (command == 3) {
                    if (scanf("%d", &amount) != 1) {
                        amount = 10;
                    }
                    
                    // Release resources
                    resource_count -= amount;
                    
                    // Clear some buffer space (bug: doesn't clear enough proportionally)
                    buffer_level -= amount / 3;
                    
                    printf("Released %d resources (remaining: %d, buffer: %d)\n",
                           amount, resource_count, buffer_level);
                }
                
                // Verify resource count remains valid
                assert(resource_count >= 0);
                
                if (command == 0 || resource_count < 10) {
                    state = STATE_IDLE;
                    printf("RELEASING -> IDLE\n");
                }
                break;
                
            default:
                printf("Invalid state!\n");
                state = STATE_IDLE;
                break;
        }
        
        // State validity check
        assert(state >= STATE_IDLE && state <= STATE_RELEASING);
        
        iteration++;
    }
    
    // Final state verification
    printf("\nFinal state: resources=%d, buffer=%d, state=%d\n",
           resource_count, buffer_level, state);
    
    assert(resource_count >= 0);
    assert(buffer_level >= 0);
    
    return 0;
}