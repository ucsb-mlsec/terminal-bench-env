#include "lockfree_stack.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <stdbool.h>

/**
 * Initialize a new lock-free stack
 * 
 * @param stack Pointer to the stack structure to initialize
 * 
 * Sets the top pointer to NULL, indicating an empty stack.
 * Uses atomic_init for proper initialization of the atomic variable.
 */
void stack_init(stack_t *stack) {
    // Initialize the atomic top pointer to NULL (empty stack)
    atomic_init(&stack->top, NULL);
}

/**
 * Push a new item onto the lock-free stack
 * 
 * @param stack Pointer to the stack
 * @param data Data value to push
 * @return true if successful, false if memory allocation failed
 * 
 * Uses a compare-and-swap (CAS) loop to atomically update the top pointer.
 * The algorithm:
 * 1. Allocate a new node
 * 2. Load the current top
 * 3. Set the new node's next to point to current top
 * 4. Try to CAS the top pointer to the new node
 * 5. If CAS fails (another thread modified top), retry from step 2
 */
bool stack_push(stack_t *stack, int data) {
    // Allocate new node
    node_t *new_node = (node_t*)malloc(sizeof(node_t));
    if (new_node == NULL) {
        return false;  // Allocation failure
    }
    
    new_node->data = data;
    
    // Load the current top pointer
    // Using memory_order_relaxed here since we only need to read the value
    // and will use stronger ordering in the CAS operation
    node_t *old_top = atomic_load_explicit(&stack->top, memory_order_relaxed);
    
    // CAS loop to push the new node
    do {
        // Set new node's next to point to current top
        new_node->next = old_top;
        
        // Try to atomically update top to point to new_node
        // If top is still old_top, update it to new_node and return true
        // If top changed (another thread modified it), old_top is updated
        // to the new value and we retry
        // Using memory_order_seq_cst for full sequential consistency
    } while (!atomic_compare_exchange_weak_explicit(
                &stack->top,
                &old_top,
                new_node,
                memory_order_seq_cst,
                memory_order_seq_cst));
    
    return true;
}

/**
 * Pop an item from the lock-free stack
 * 
 * @param stack Pointer to the stack
 * @param data Pointer to store the popped data
 * @return true if an item was popped, false if stack was empty
 * 
 * Uses a compare-and-swap (CAS) loop to atomically update the top pointer.
 * The algorithm:
 * 1. Load the current top
 * 2. If top is NULL, stack is empty, return false
 * 3. Load the next pointer from the top node
 * 4. Try to CAS the top pointer to point to next
 * 5. If CAS fails (another thread modified top), retry from step 1
 * 6. If CAS succeeds, retrieve data and free the old top node
 * 
 * NOTE: This implementation is vulnerable to the ABA problem!
 * If thread T1 reads top (A), gets preempted, thread T2 pops A and B,
 * then pushes A back, T1's CAS will succeed even though the stack
 * state has changed. This can lead to corrupted pointers.
 */
bool stack_pop(stack_t *stack, int *data) {
    // Load the current top pointer
    // Using memory_order_relaxed - BUG: should use stronger ordering
    // to ensure proper synchronization with push operations
    node_t *old_top = atomic_load_explicit(&stack->top, memory_order_relaxed);
    
    // CAS loop to pop the top node
    while (old_top != NULL) {
        // Read the next pointer from the node we're trying to pop
        node_t *next = old_top->next;
        
        // Try to atomically update top to point to next
        // BUG: This is vulnerable to the ABA problem!
        // We're only comparing pointers, not using any version counter
        // or other ABA protection mechanism
        if (atomic_compare_exchange_weak_explicit(
                &stack->top,
                &old_top,
                next,
                memory_order_relaxed,  // BUG: Using relaxed ordering
                memory_order_relaxed)) {
            
            // CAS succeeded - we've popped the node
            *data = old_top->data;
            free(old_top);
            return true;
        }
        
        // CAS failed - old_top has been updated to current value, retry
    }
    
    // Stack was empty
    return false;
}