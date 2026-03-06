# Lock-Free Queue System

## Overview

This is a concurrent message processing system designed to handle high-throughput scenarios using a lock-free queue implementation. The system avoids traditional mutex-based synchronization to eliminate lock contention overhead and improve scalability across multiple threads.

The lock-free queue allows multiple producer and consumer threads to operate concurrently without blocking, using atomic operations to maintain thread safety and data consistency.

## Architecture

### Lock-Free Queue Design

The queue implementation uses atomic compare-and-swap (CAS) operations to enable thread-safe enqueue and dequeue operations without locks. Key design elements include:

- **Node-based structure**: Queue uses dynamically allocated nodes containing data and next pointers
- **Head and tail pointers**: Separate atomic pointers for dequeue (head) and enqueue (tail) operations
- **Atomic operations**: All pointer updates use atomic CAS to prevent race conditions
- **Memory ordering**: Carefully chosen memory ordering semantics ensure visibility of updates across threads

### Thread Model

The system supports:
- Multiple concurrent producer threads performing enqueue operations
- Multiple concurrent consumer threads performing dequeue operations
- Lock-free progress guarantee: at least one thread makes progress even under contention

## Data Structures

### Queue Node
```c
typedef struct node {
    int value;
    struct node* next;
} node_t;
```

### Queue Structure
```c
typedef struct {
    node_t* head;
    node_t* tail;
} lockfree_queue_t;
```

## API

### Initialization
```c
void queue_init(lockfree_queue_t* queue);
```
Initializes an empty queue with a dummy node.

### Enqueue
```c
void enqueue(lockfree_queue_t* queue, int value);
```
Adds a value to the tail of the queue. Thread-safe for concurrent producers.

### Dequeue
```c
int dequeue(lockfree_queue_t* queue, int* value);
```
Removes and returns a value from the head of the queue. Returns 1 on success, 0 if queue is empty. Thread-safe for concurrent consumers.

### Cleanup
```c
void queue_destroy(lockfree_queue_t* queue);
```
Frees all remaining nodes in the queue.

## Testing

### Test Harness

The test harness (`test_harness.c`) validates the queue implementation under concurrent load:

- **4 Producer Threads**: Each producer enqueues 10,000 sequential integer messages (IDs 0-9,999)
- **4 Consumer Threads**: Each consumer dequeues messages and tracks received values
- **Total Messages**: 40,000 messages flow through the system
- **Validation**: After all threads complete, the system verifies that all values were received exactly once

### Test Methodology

1. Spawn 4 producer threads, each with a unique ID range
2. Spawn 4 consumer threads that compete for messages
3. Producers enqueue their assigned range of integers
4. Consumers dequeue and mark received values in a validation bitmap
5. After all threads join, check for missing or duplicate messages

### Stress Testing

The test is designed to expose race conditions by:
- Creating high contention with multiple concurrent threads
- Using a large message count to increase probability of edge cases
- Running producers and consumers simultaneously
- Validating complete correctness of all operations

## Expected Behavior

Under correct implementation, the system must guarantee:

1. **Completeness**: All 40,000 messages must be received by consumers
2. **Uniqueness**: Each message value must be received exactly once (no duplicates)
3. **Memory Safety**: No segmentation faults, use-after-free, or memory leaks
4. **Thread Safety**: All threads must complete successfully without deadlock or corruption
5. **Atomicity**: Each enqueue and dequeue operation must be atomic from the perspective of other threads

## Building

### Compilation

Build the test harness and queue implementation:

```bash
make
```

This compiles with appropriate flags including:
- `-pthread` for POSIX threads support
- `-std=c11` for C11 atomic operations
- `-O2` for optimization
- `-Wall -Wextra` for comprehensive warnings

### Clean Build

Remove compiled artifacts:

```bash
make clean
```

### Running Tests

Execute the test harness:

```bash
./test_harness
```

The test will output progress messages and final validation results.

## Success Criteria

A successful test run must satisfy all of the following:

1. **All 40,000 messages accounted for**: No missing values in the validation check
2. **No duplicate messages**: Each value from 0-9,999 received exactly once from each producer
3. **No segmentation faults**: No memory access violations or crashes
4. **Clean thread completion**: All producer and consumer threads exit normally
5. **Zero memory errors**: No leaks, double-frees, or invalid accesses (verifiable with valgrind)

### Success Output Example

```
Starting producers...
Starting consumers...
All producers finished
All consumers finished
Validation: PASS - All 40000 messages received exactly once
```

### Failure Indicators

Signs of queue implementation bugs:
- "Missing messages" errors
- "Duplicate messages" errors
- Segmentation fault crashes
- Threads hanging indefinitely
- Valgrind reporting memory errors

## Troubleshooting

### Common Issues

**Race Conditions**: Most failures stem from improper synchronization in CAS loops or incorrect memory ordering.

**ABA Problem**: Lock-free algorithms are susceptible to the ABA problem where a pointer is changed from A to B and back to A, causing CAS to succeed incorrectly.

**Memory Ordering**: Incorrect memory ordering can lead to visibility issues where one thread doesn't see updates from another.

**Node Recycling**: Premature freeing of nodes can cause use-after-free bugs when other threads still hold references.

## Performance Characteristics

Expected performance properties:
- **Lock-free progress**: System continues making progress even under high contention
- **Scalability**: Performance should improve with more threads (up to core count)
- **Low latency**: Individual operations complete quickly without blocking
- **High throughput**: Can process millions of messages per second on modern hardware

## Implementation Notes

The queue uses Michael-Scott style lock-free queue algorithm principles, adapted for this specific use case. Proper implementation requires careful attention to:
- Atomic operation ordering
- Prevention of premature node reclamation
- Correct handling of concurrent enqueue/dequeue operations
- Proper initialization and dummy node management