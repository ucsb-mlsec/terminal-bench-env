#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "utils.h"

#define MAX_BUFFER_SIZE 1024
#define PROCESSOR_VERSION "1.0"

typedef struct {
    char *data;
    int size;
    int capacity;
} DataBuffer;

typedef struct {
    int items_processed;
    int errors;
    DataBuffer buffer;
} ProcessorState;

/* Initialize the processor module */
int init_processor(void);

/* Process data from input to output */
int process_data(const char *input, char *output, int max_len);

/* Cleanup and free processor resources */
void cleanup_processor(void);

/* Get current processor state */
ProcessorState* get_processor_state(void);

#endif /* PROCESSOR_H */