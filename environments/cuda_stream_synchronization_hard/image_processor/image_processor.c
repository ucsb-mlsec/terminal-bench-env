#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <math.h>

#define MAX_PATH_LENGTH 512
#define MAX_IMAGES 100
#define NUM_THREADS 6
#define WORK_QUEUE_SIZE 100

// Image structure for storing PGM image data
typedef struct {
    int width;
    int height;
    int max_value;
    unsigned char *data;
    char input_path[MAX_PATH_LENGTH];
    char output_path[MAX_PATH_LENGTH];
} Image;

// Work item structure for the queue
typedef struct {
    Image *image;
    int valid;
} WorkItem;

// Work queue structure
typedef struct {
    WorkItem items[WORK_QUEUE_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int shutdown;
} WorkQueue;

// Thread pool structure
typedef struct {
    pthread_t threads[NUM_THREADS];
    WorkQueue *queue;
    int num_threads;
} ThreadPool;

// Global statistics
int images_processed = 0;
int images_failed = 0;
pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;

// Shared temporary buffer for edge detection (BUG: no synchronization)
unsigned char *temp_buffer = NULL;
int temp_buffer_size = 0;

// Global image list for tracking
Image *global_image_list[MAX_IMAGES];
int global_image_count = 0;

// Function prototypes
Image* create_image(int width, int height, int max_value);
void free_image(Image *img);
Image* read_pgm_image(const char *filename);
int write_pgm_image(const char *filename, Image *img);
void apply_edge_detection(Image *img);
WorkQueue* create_work_queue();
void destroy_work_queue(WorkQueue *queue);
int enqueue_work(WorkQueue *queue, Image *img);
Image* dequeue_work(WorkQueue *queue);
void* worker_thread(void *arg);
ThreadPool* create_thread_pool(int num_threads);
void destroy_thread_pool(ThreadPool *pool);
int process_directory(const char *input_dir, const char *output_dir, ThreadPool *pool);
void print_usage(const char *program_name);

// Create a new image structure
Image* create_image(int width, int height, int max_value) {
    Image *img = (Image*)malloc(sizeof(Image));
    if (!img) {
        fprintf(stderr, "Failed to allocate image structure\n");
        return NULL;
    }
    
    img->width = width;
    img->height = height;
    img->max_value = max_value;
    img->data = (unsigned char*)calloc(width * height, sizeof(unsigned char));
    
    if (!img->data) {
        fprintf(stderr, "Failed to allocate image data\n");
        free(img);
        return NULL;
    }
    
    memset(img->input_path, 0, MAX_PATH_LENGTH);
    memset(img->output_path, 0, MAX_PATH_LENGTH);
    
    return img;
}

// Free image memory
void free_image(Image *img) {
    if (img) {
        if (img->data) {
            free(img->data);
        }
        free(img);
    }
}

// Read PGM P2 (ASCII) format image
Image* read_pgm_image(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P2") != 0) {
        fprintf(stderr, "Invalid PGM format (expected P2): %s\n", filename);
        fclose(fp);
        return NULL;
    }
    
    // Skip comments
    int c;
    while ((c = fgetc(fp)) == '#') {
        while ((c = fgetc(fp)) != '\n' && c != EOF);
    }
    ungetc(c, fp);
    
    int width, height, max_value;
    if (fscanf(fp, "%d %d", &width, &height) != 2) {
        fprintf(stderr, "Failed to read image dimensions: %s\n", filename);
        fclose(fp);
        return NULL;
    }
    
    if (fscanf(fp, "%d", &max_value) != 1) {
        fprintf(stderr, "Failed to read max value: %s\n", filename);
        fclose(fp);
        return NULL;
    }
    
    Image *img = create_image(width, height, max_value);
    if (!img) {
        fclose(fp);
        return NULL;
    }
    
    // Read pixel data
    for (int i = 0; i < width * height; i++) {
        int pixel;
        if (fscanf(fp, "%d", &pixel) != 1) {
            fprintf(stderr, "Failed to read pixel data: %s\n", filename);
            free_image(img);
            fclose(fp);
            return NULL;
        }
        img->data[i] = (unsigned char)pixel;
    }
    
    fclose(fp);
    strncpy(img->input_path, filename, MAX_PATH_LENGTH - 1);
    
    return img;
}

// Write PGM P2 (ASCII) format image
int write_pgm_image(const char *filename, Image *img) {
    if (!img || !img->data) {
        fprintf(stderr, "Invalid image\n");
        return -1;
    }
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to create output file: %s\n", filename);
        return -1;
    }
    
    // Write header
    fprintf(fp, "P2\n");
    fprintf(fp, "# Created by image processor\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max_value);
    
    // Write pixel data
    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            fprintf(fp, "%d", img->data[i * img->width + j]);
            if (j < img->width - 1) {
                fprintf(fp, " ");
            }
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    return 0;
}

// Apply Sobel edge detection filter
void apply_edge_detection(Image *img) {
    if (!img || !img->data) {
        return;
    }
    
    int width = img->width;
    int height = img->height;
    int size = width * height;
    
    // BUG: Using shared global buffer without synchronization
    // This causes race conditions when multiple threads process images simultaneously
    if (temp_buffer_size < size) {
        if (temp_buffer) {
            free(temp_buffer);
        }
        temp_buffer = (unsigned char*)malloc(size);
        temp_buffer_size = size;
    }
    
    // Copy original data to temp buffer
    memcpy(temp_buffer, img->data, size);
    
    // Sobel kernels
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    // Apply Sobel operator
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum_x = 0;
            int sum_y = 0;
            
            // Convolve with kernels
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = temp_buffer[(y + ky) * width + (x + kx)];
                    sum_x += pixel * gx[ky + 1][kx + 1];
                    sum_y += pixel * gy[ky + 1][kx + 1];
                }
            }
            
            // Calculate gradient magnitude
            int magnitude = (int)sqrt(sum_x * sum_x + sum_y * sum_y);
            
            // Clamp to max value
            if (magnitude > img->max_value) {
                magnitude = img->max_value;
            }
            
            img->data[y * width + x] = (unsigned char)magnitude;
        }
    }
    
    // Handle borders (set to 0)
    for (int x = 0; x < width; x++) {
        img->data[x] = 0;
        img->data[(height - 1) * width + x] = 0;
    }
    for (int y = 0; y < height; y++) {
        img->data[y * width] = 0;
        img->data[y * width + (width - 1)] = 0;
    }
    
    // Simulate some processing time
    usleep(1000);
}

// Create work queue
WorkQueue* create_work_queue() {
    WorkQueue *queue = (WorkQueue*)malloc(sizeof(WorkQueue));
    if (!queue) {
        fprintf(stderr, "Failed to allocate work queue\n");
        return NULL;
    }
    
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    queue->shutdown = 0;
    
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
    
    for (int i = 0; i < WORK_QUEUE_SIZE; i++) {
        queue->items[i].image = NULL;
        queue->items[i].valid = 0;
    }
    
    return queue;
}

// Destroy work queue
void destroy_work_queue(WorkQueue *queue) {
    if (!queue) {
        return;
    }
    
    pthread_mutex_lock(&queue->mutex);
    queue->shutdown = 1;
    pthread_cond_broadcast(&queue->not_empty);
    pthread_cond_broadcast(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    
    free(queue);
}

// Enqueue work item
int enqueue_work(WorkQueue *queue, Image *img) {
    if (!queue || !img) {
        return -1;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count >= WORK_QUEUE_SIZE && !queue->shutdown) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    
    queue->items[queue->tail].image = img;
    queue->items[queue->tail].valid = 1;
    queue->tail = (queue->tail + 1) % WORK_QUEUE_SIZE;
    queue->count++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return 0;
}

// Dequeue work item
Image* dequeue_work(WorkQueue *queue) {
    if (!queue) {
        return NULL;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count == 0 && !queue->shutdown) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    if (queue->count == 0 && queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }
    
    Image *img = queue->items[queue->head].image;
    queue->items[queue->head].valid = 0;
    queue->items[queue->head].image = NULL;
    queue->head = (queue->head + 1) % WORK_QUEUE_SIZE;
    queue->count--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return img;
}

// Worker thread function
void* worker_thread(void *arg) {
    WorkQueue *queue = (WorkQueue*)arg;
    
    while (1) {
        Image *img = dequeue_work(queue);
        
        if (!img) {
            // Queue is shutting down
            break;
        }
        
        // Process the image
        apply_edge_detection(img);
        
        // Write output
        if (write_pgm_image(img->output_path, img) == 0) {
            // BUG: Updating shared counter without proper synchronization
            images_processed++;
        } else {
            // BUG: Updating shared counter without proper synchronization
            images_failed++;
        }
        
        // BUG: Race condition in cleanup - image might be freed while another thread
        // is still accessing global_image_list
        free_image(img);
    }
    
    return NULL;
}

// Create thread pool
ThreadPool* create_thread_pool(int num_threads) {
    ThreadPool *pool = (ThreadPool*)malloc(sizeof(ThreadPool));
    if (!pool) {
        fprintf(stderr, "Failed to allocate thread pool\n");
        return NULL;
    }
    
    pool->num_threads = num_threads;
    pool->queue = create_work_queue();
    
    if (!pool->queue) {
        free(pool);
        return NULL;
    }
    
    // Create worker threads
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread, pool->queue) != 0) {
            fprintf(stderr, "Failed to create worker thread %d\n", i);
            destroy_work_queue(pool->queue);
            free(pool);
            return NULL;
        }
    }
    
    return pool;
}

// Destroy thread pool
void destroy_thread_pool(ThreadPool *pool) {
    if (!pool) {
        return;
    }
    
    // Signal shutdown
    pthread_mutex_lock(&pool->queue->mutex);
    pool->queue->shutdown = 1;
    pthread_cond_broadcast(&pool->queue->not_empty);
    pthread_mutex_unlock(&pool->queue->mutex);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    destroy_work_queue(pool->queue);
    free(pool);
}

// Check if file is a PGM image
int is_pgm_file(const char *filename) {
    size_t len = strlen(filename);
    if (len < 4) {
        return 0;
    }
    return (strcmp(filename + len - 4, ".pgm") == 0);
}

// Process all images in directory
int process_directory(const char *input_dir, const char *output_dir, ThreadPool *pool) {
    DIR *dir = opendir(input_dir);
    if (!dir) {
        fprintf(stderr, "Failed to open input directory: %s\n", input_dir);
        return -1;
    }
    
    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        if (mkdir(output_dir, 0755) != 0) {
            fprintf(stderr, "Failed to create output directory: %s\n", output_dir);
            closedir(dir);
            return -1;
        }
    }
    
    struct dirent *entry;
    int total_images = 0;
    
    printf("Scanning directory: %s\n", input_dir);
    
    while ((entry = readdir(dir)) != NULL) {
        if (!is_pgm_file(entry->d_name)) {
            continue;
        }
        
        char input_path[MAX_PATH_LENGTH];
        char output_path[MAX_PATH_LENGTH];
        
        snprintf(input_path, MAX_PATH_LENGTH, "%s/%s", input_dir, entry->d_name);
        snprintf(output_path, MAX_PATH_LENGTH, "%s/%s", output_dir, entry->d_name);
        
        printf("Reading image: %s\n", entry->d_name);
        
        Image *img = read_pgm_image(input_path);
        if (!img) {
            fprintf(stderr, "Failed to read image: %s\n", input_path);
            continue;
        }
        
        strncpy(img->output_path, output_path, MAX_PATH_LENGTH - 1);
        
        // Add to global list (BUG: no synchronization for global_image_list access)
        if (global_image_count < MAX_IMAGES) {
            global_image_list[global_image_count] = img;
            global_image_count++;
        }
        
        // Enqueue for processing
        if (enqueue_work(pool->queue, img) == 0) {
            total_images++;
        } else {
            fprintf(stderr, "Failed to enqueue image: %s\n", entry->d_name);
            free_image(img);
        }
    }
    
    closedir(dir);
    
    printf("Total images queued: %d\n", total_images);
    
    return total_images;
}

// Print usage information
void print_usage(const char *program_name) {
    printf("Usage: %s <input_directory> <output_directory>\n", program_name);
    printf("\n");
    printf("Process PGM images with edge detection filter.\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  input_directory   Directory containing input PGM images\n");
    printf("  output_directory  Directory where processed images will be saved\n");
    printf("\n");
    printf("The program uses %d worker threads for parallel processing.\n", NUM_THREADS);
}

// Main function
int main(int argc, char *argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *input_dir = argv[1];
    const char *output_dir = argv[2];
    
    printf("=== Multi-threaded Image Processor ===\n");
    printf("Input directory:  %s\n", input_dir);
    printf("Output directory: %s\n", output_dir);
    printf("Worker threads:   %d\n", NUM_THREADS);
    printf("\n");
    
    // Create thread pool
    ThreadPool *pool = create_thread_pool(NUM_THREADS);
    if (!pool) {
        fprintf(stderr, "Failed to create thread pool\n");
        return 1;
    }
    
    // Process all images in directory
    int total = process_directory(input_dir, output_dir, pool);
    
    if (total <= 0) {
        fprintf(stderr, "No images were processed\n");
        destroy_thread_pool(pool);
        return 1;
    }
    
    // Wait a bit for processing to complete
    printf("\nProcessing images...\n");
    sleep(2);
    
    // Shutdown thread pool
    destroy_thread_pool(pool);
    
    // Print statistics
    printf("\n=== Processing Complete ===\n");
    printf("Images processed successfully: %d\n", images_processed);
    printf("Images failed: %d\n", images_failed);
    printf("Total images: %d\n", total);
    
    // Cleanup global resources
    if (temp_buffer) {
        free(temp_buffer);
    }
    
    // Note: Not freeing global_image_list entries because they were already
    // freed by worker threads (potential use-after-free bug scenario)
    
    return 0;
}