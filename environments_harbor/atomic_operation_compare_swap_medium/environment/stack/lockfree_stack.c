#include <stdlib.h>
#include <stdatomic.h>
#include "lockfree_stack.h"

typedef struct node {
    void* data;
    struct node* next;
} node_t;

static atomic_uintptr_t top = 0;

void push(void* data) {
    node_t* new_node = malloc(sizeof(node_t));
    new_node->data = data;
    
    node_t* old_top;
    do {
        old_top = (node_t*)atomic_load(&top);
        new_node->next = old_top;
    } while (!atomic_compare_exchange_weak(&top, (uintptr_t*)&old_top, (uintptr_t)new_node));
}

void* pop(void) {
    node_t* old_top;
    node_t* new_top;
    
    do {
        old_top = (node_t*)atomic_load(&top);
        if (old_top == NULL) return NULL;
        new_top = old_top->next;
    } while (!atomic_compare_exchange_weak(&top, (uintptr_t*)&old_top, (uintptr_t)new_top));
    
    void* data = old_top->data;
    free(old_top);
    return data;
}