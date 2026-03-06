#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define NUM_WORKERS 4
#define NUM_TASKS 100
#define MAX_QUEUE_SIZE 50

// Task structure with varying computational intensity
typedef struct {
    int task_id;
    int execution_time_ms;
} Task;

// Node for linked list queue
typedef struct TaskNode {
    Task task;
    struct TaskNode* next;
} TaskNode;

// Per-worker queue structure
typedef struct {
    TaskNode* head;
    TaskNode* tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int finished;
} WorkerQueue;

// Global data structures
WorkerQueue worker_queues[NUM_WORKERS];
pthread_t worker_threads[NUM_WORKERS];
int tasks_completed[NUM_WORKERS] = {0};
int total_tasks_completed = 0;
pthread_mutex_t completion_mutex = PTHREAD_MUTEX_INITIALIZER;

// Initialize a worker queue
void init_queue(WorkerQueue* queue) {
    queue->head = NULL;
    queue->tail = NULL;
    queue->count = 0;
    queue->finished = 0;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->cond, NULL);
}

// Enqueue a task to a worker's queue
void enqueue_task(WorkerQueue* queue, Task task) {
    TaskNode* node = (TaskNode*)malloc(sizeof(TaskNode));
    node->task = task;
    node->next = NULL;
    
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->tail == NULL) {
        queue->head = node;
        queue->tail = node;
    } else {
        queue->tail->next = node;
        queue->tail = node;
    }
    queue->count++;
    
    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
}

// Dequeue a task from a worker's queue
int dequeue_task(WorkerQueue* queue, Task* task) {
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->head == NULL && !queue->finished) {
        pthread_cond_wait(&queue->cond, &queue->mutex);
    }
    
    if (queue->head == NULL) {
        pthread_mutex_unlock(&queue->mutex);
        return 0;
    }
    
    TaskNode* node = queue->head;
    *task = node->task;
    queue->head = node->next;
    
    if (queue->head == NULL) {
        queue->tail = NULL;
    }
    
    queue->count--;
    free(node);
    
    pthread_mutex_unlock(&queue->mutex);
    return 1;
}

// Simulate task execution
void execute_task(Task task) {
    // Simulate computational work by sleeping
    usleep(task.execution_time_ms * 1000);
}

// Worker thread function - processes tasks from own queue only (NO work stealing)
void* worker_function(void* arg) {
    int worker_id = *(int*)arg;
    free(arg);
    
    WorkerQueue* my_queue = &worker_queues[worker_id];
    
    while (1) {
        Task task;
        
        // Try to get a task from own queue only
        if (dequeue_task(my_queue, &task)) {
            // Execute the task
            execute_task(task);
            
            // Increment completion counter
            pthread_mutex_lock(&completion_mutex);
            tasks_completed[worker_id]++;
            total_tasks_completed++;
            pthread_mutex_unlock(&completion_mutex);
            
            printf("Worker %d completed task %d (execution time: %dms)\n", 
                   worker_id, task.task_id, task.execution_time_ms);
        } else {
            // No more tasks in queue and finished flag is set
            break;
        }
    }
    
    printf("Worker %d finished\n", worker_id);
    return NULL;
}

// Generate tasks with varying execution times to create imbalance
void generate_tasks(Task* tasks) {
    // Create imbalanced workload:
    // Tasks 0-24: light (10-30ms)
    // Tasks 25-49: medium (50-100ms)
    // Tasks 50-74: heavy (150-200ms)
    // Tasks 75-99: light (10-30ms)
    
    for (int i = 0; i < NUM_TASKS; i++) {
        tasks[i].task_id = i;
        
        if (i < 25) {
            // Light tasks
            tasks[i].execution_time_ms = 10 + (rand() % 21);
        } else if (i < 50) {
            // Medium tasks
            tasks[i].execution_time_ms = 50 + (rand() % 51);
        } else if (i < 75) {
            // Heavy tasks
            tasks[i].execution_time_ms = 150 + (rand() % 51);
        } else {
            // Light tasks
            tasks[i].execution_time_ms = 10 + (rand() % 21);
        }
    }
}

// Assign tasks using round-robin (causes severe imbalance)
void assign_tasks_round_robin(Task* tasks) {
    printf("Assigning tasks using round-robin approach...\n");
    
    for (int i = 0; i < NUM_TASKS; i++) {
        int worker_id = i % NUM_WORKERS;
        enqueue_task(&worker_queues[worker_id], tasks[i]);
    }
    
    printf("Task assignment complete. Each worker has 25 tasks.\n");
}

int main() {
    srand(time(NULL));
    
    printf("Starting broken task scheduler benchmark...\n");
    printf("Workers: %d, Tasks: %d\n\n", NUM_WORKERS, NUM_TASKS);
    
    // Record start time
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Initialize worker queues
    for (int i = 0; i < NUM_WORKERS; i++) {
        init_queue(&worker_queues[i]);
    }
    
    // Generate tasks with varying execution times
    Task tasks[NUM_TASKS];
    generate_tasks(tasks);
    
    // Assign tasks to workers using round-robin (creates imbalance)
    assign_tasks_round_robin(tasks);
    
    // Create worker threads
    for (int i = 0; i < NUM_WORKERS; i++) {
        int* worker_id = malloc(sizeof(int));
        *worker_id = i;
        pthread_create(&worker_threads[i], NULL, worker_function, worker_id);
    }
    
    // Wait for all workers to complete
    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_mutex_lock(&worker_queues[i].mutex);
        worker_queues[i].finished = 1;
        pthread_cond_signal(&worker_queues[i].cond);
        pthread_mutex_unlock(&worker_queues[i].mutex);
    }
    
    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_join(worker_threads[i], NULL);
    }
    
    // Record end time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate execution time
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    // Print results
    printf("\n========== BENCHMARK RESULTS ==========\n");
    printf("Total execution time: %.2f seconds\n", elapsed);
    printf("Total tasks completed: %d\n", total_tasks_completed);
    printf("\nPer-worker task distribution:\n");
    
    for (int i = 0; i < NUM_WORKERS; i++) {
        printf("  Worker %d: %d tasks completed\n", i, tasks_completed[i]);
    }
    
    printf("\nNOTE: This is the baseline implementation with round-robin scheduling.\n");
    printf("The load imbalance causes poor performance.\n");
    printf("Some workers finish early while others are overloaded.\n");
    printf("=======================================\n");
    
    // Cleanup
    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_mutex_destroy(&worker_queues[i].mutex);
        pthread_cond_destroy(&worker_queues[i].cond);
    }
    pthread_mutex_destroy(&completion_mutex);
    
    return 0;
}