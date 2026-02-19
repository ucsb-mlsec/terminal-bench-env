#ifndef LOCKFREE_QUEUE_H
#define LOCKFREE_QUEUE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>

typedef struct node {
    int data;
    _Atomic(struct node*) next;
} node_t;

typedef struct queue {
    _Atomic(node_t*) head;
    _Atomic(node_t*) tail;
} queue_t;

void queue_init(queue_t* queue);
void queue_enqueue(queue_t* queue, int data);
bool queue_dequeue(queue_t* queue, int* data);
void queue_destroy(queue_t* queue);

#endif