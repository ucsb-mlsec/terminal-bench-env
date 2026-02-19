#ifndef LOCKFREE_STACK_H
#define LOCKFREE_STACK_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

/* Node structure for the lock-free stack */
typedef struct stack_node {
    void* data;
    struct stack_node* _Atomic next;
} stack_node_t;

/* Lock-free stack structure */
typedef struct {
    stack_node_t* _Atomic top;
} stack_t;

/* Initialize a new stack */
void stack_init(stack_t* stack);

/* Push an item onto the stack
 * Returns true on success, false on failure (e.g., allocation failure)
 */
bool stack_push(stack_t* stack, void* data);

/* Pop an item from the stack
 * Returns the data pointer, or NULL if the stack is empty
 */
void* stack_pop(stack_t* stack);

#endif /* LOCKFREE_STACK_H */