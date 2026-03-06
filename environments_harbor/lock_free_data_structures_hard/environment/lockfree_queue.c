#include "lockfree_queue.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdbool.h>

typedef struct queue_node {
    void *data;
    _Atomic(struct queue_node*) next;
} queue_node_t;

struct lockfree_queue {
    _Atomic(queue_node_t*) head;
    _Atomic(queue_node_t*) tail;
};

lockfree_queue_t* queue_init(void) {
    lockfree_queue_t *queue = malloc(sizeof(lockfree_queue_t));
    if (!queue) {
        return NULL;
    }
    
    // Create sentinel/dummy node
    queue_node_t *sentinel = malloc(sizeof(queue_node_t));
    if (!sentinel) {
        free(queue);
        return NULL;
    }
    
    sentinel->data = NULL;
    atomic_store(&sentinel->next, NULL);
    
    atomic_store(&queue->head, sentinel);
    atomic_store(&queue->tail, sentinel);
    
    return queue;
}

bool queue_enqueue(lockfree_queue_t *queue, void *data) {
    if (!queue || !data) {
        return false;
    }
    
    queue_node_t *node = malloc(sizeof(queue_node_t));
    if (!node) {
        return false;
    }
    
    node->data = data;
    atomic_store(&node->next, NULL);
    
    while (true) {
        queue_node_t *tail = atomic_load(&queue->tail);
        queue_node_t *next = atomic_load(&tail->next);
        
        // Check if tail is still consistent
        if (tail == atomic_load(&queue->tail)) {
            if (next == NULL) {
                // Try to link node at the end
                if (atomic_compare_exchange_weak(&tail->next, &next, node)) {
                    // Enqueue is done, try to swing tail to the inserted node
                    atomic_compare_exchange_weak(&queue->tail, &tail, node);
                    return true;
                }
            } else {
                // Tail was not pointing to the last node, try to swing tail
                atomic_compare_exchange_weak(&queue->tail, &tail, next);
            }
        }
    }
}

bool queue_dequeue(lockfree_queue_t *queue, void **data) {
    if (!queue || !data) {
        return false;
    }
    
    while (true) {
        queue_node_t *head = atomic_load(&queue->head);
        queue_node_t *tail = atomic_load(&queue->tail);
        queue_node_t *next = atomic_load(&head->next);
        
        // BUG: Using relaxed memory order here creates a race condition
        // The load of head->next with memory_order_relaxed means we might
        // see an inconsistent view of the queue state across threads
        // This should use memory_order_acquire to ensure proper synchronization
        next = atomic_load_explicit(&head->next, memory_order_relaxed);
        
        // Check if head is still consistent
        if (head == atomic_load(&queue->head)) {
            if (head == tail) {
                if (next == NULL) {
                    // Queue is empty
                    return false;
                }
                // Tail is falling behind, try to advance it
                atomic_compare_exchange_weak(&queue->tail, &tail, next);
            } else {
                if (next == NULL) {
                    // This should not happen in a correct implementation
                    continue;
                }
                
                // Read value before CAS, otherwise another dequeue might free the node
                *data = next->data;
                
                // BUG: The CAS operation doesn't use proper memory ordering
                // This creates a race where multiple threads can dequeue the same element
                // or miss updates from other threads due to relaxed semantics
                if (atomic_compare_exchange_weak_explicit(&queue->head, &head, next,
                                                          memory_order_relaxed,
                                                          memory_order_relaxed)) {
                    // Successfully dequeued
                    free(head);
                    return true;
                }
            }
        }
    }
}

void queue_destroy(lockfree_queue_t *queue) {
    if (!queue) {
        return;
    }
    
    // Drain the queue
    void *data;
    while (queue_dequeue(queue, &data)) {
        // Just remove elements, caller should have ownership of data
    }
    
    // Free the sentinel node
    queue_node_t *sentinel = atomic_load(&queue->head);
    free(sentinel);
    
    free(queue);
}

bool queue_is_empty(lockfree_queue_t *queue) {
    if (!queue) {
        return true;
    }
    
    queue_node_t *head = atomic_load(&queue->head);
    queue_node_t *next = atomic_load(&head->next);
    
    return next == NULL;
}