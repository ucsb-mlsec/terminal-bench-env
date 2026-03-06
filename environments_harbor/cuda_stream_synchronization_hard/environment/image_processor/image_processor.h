#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <pthread.h>
#include <stdint.h>

#define MAX_THREADS 8
#define QUEUE_SIZE 100
#define MAX_FILENAME 256

/* Image structure for storing grayscale image data */
typedef struct {
    int width;
    int height;
    int max_value;
    unsigned char* data;
} Image;

/* Work item structure for job queue */
typedef struct {
    char input_filename[MAX_FILENAME];
    char output_filename[MAX_FILENAME];
    int status;  /* 0 = pending, 1 = processing, 2 = complete, -1 = error */
} WorkItem;

/* Work queue structure */
typedef struct {
    WorkItem items[QUEUE_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int shutdown;
} WorkQueue;

/* Thread pool structure */
typedef struct {
    pthread_t threads[MAX_THREADS];
    int num_threads;
    WorkQueue* queue;
    int active;
} ThreadPool;

/* Image processing functions */
Image* image_load(char* filename);
int image_save(Image* img, char* filename);
void image_free(Image* img);
Image* image_create(int width, int height, int max_value);
Image* image_copy(Image* src);

/* Edge detection filter */
void apply_edge_detection(Image* img);

/* Batch processing */
int process_image_batch(char* input_dir, char* output_dir, int num_threads);

/* Work queue operations */
WorkQueue* queue_create(void);
void queue_destroy(WorkQueue* queue);
int queue_push(WorkQueue* queue, WorkItem* item);
int queue_pop(WorkQueue* queue, WorkItem* item);

/* Thread pool operations */
ThreadPool* threadpool_create(int num_threads, WorkQueue* queue);
void threadpool_destroy(ThreadPool* pool);
void* worker_thread(void* arg);

/* Statistics tracking */
typedef struct {
    int images_processed;
    int images_failed;
    double total_time;
} ProcessingStats;

extern ProcessingStats global_stats;

#endif /* IMAGE_PROCESSOR_H */