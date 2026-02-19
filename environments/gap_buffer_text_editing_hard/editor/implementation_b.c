#include "text_buffer.h"
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 1024
#define GAP_SIZE 512

struct TextBuffer {
    char* data;
    int gap_start;
    int gap_end;
    int capacity;
    int length;
};

TextBuffer* text_buffer_create(void) {
    TextBuffer* buffer = (TextBuffer*)malloc(sizeof(TextBuffer));
    if (!buffer) return NULL;
    
    buffer->capacity = INITIAL_CAPACITY;
    buffer->data = (char*)malloc(buffer->capacity);
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    
    buffer->gap_start = 0;
    buffer->gap_end = buffer->capacity;
    buffer->length = 0;
    
    return buffer;
}

void text_buffer_destroy(TextBuffer* buffer) {
    if (buffer) {
        free(buffer->data);
        free(buffer);
    }
}

static void move_gap(TextBuffer* buffer, int position) {
    if (position == buffer->gap_start) {
        return;
    }
    
    if (position < buffer->gap_start) {
        int move_size = buffer->gap_start - position;
        memmove(buffer->data + buffer->gap_end - move_size,
                buffer->data + position,
                move_size);
        buffer->gap_end -= move_size;
        buffer->gap_start = position;
    } else {
        int move_size = position - buffer->gap_start;
        memmove(buffer->data + buffer->gap_start,
                buffer->data + buffer->gap_end,
                move_size);
        buffer->gap_start += move_size;
        buffer->gap_end += move_size;
    }
}

static void expand_gap(TextBuffer* buffer) {
    int new_capacity = buffer->capacity * 2;
    char* new_data = (char*)malloc(new_capacity);
    if (!new_data) return;
    
    memcpy(new_data, buffer->data, buffer->gap_start);
    
    int text_after_gap = buffer->capacity - buffer->gap_end;
    int new_gap_end = new_capacity - text_after_gap;
    memcpy(new_data + new_gap_end, buffer->data + buffer->gap_end, text_after_gap);
    
    free(buffer->data);
    buffer->data = new_data;
    buffer->gap_end = new_gap_end;
    buffer->capacity = new_capacity;
}

void text_buffer_insert(TextBuffer* buffer, int position, char c) {
    if (!buffer || position < 0 || position > buffer->length) return;
    
    move_gap(buffer, position);
    
    if (buffer->gap_start >= buffer->gap_end) {
        expand_gap(buffer);
    }
    
    buffer->data[buffer->gap_start] = c;
    buffer->gap_start++;
    buffer->length++;
}

void text_buffer_delete(TextBuffer* buffer, int position) {
    if (!buffer || position < 0 || position >= buffer->length) return;
    
    move_gap(buffer, position);
    
    if (buffer->gap_end < buffer->capacity) {
        buffer->gap_end++;
        buffer->length--;
    }
}

char* text_buffer_get_text(TextBuffer* buffer) {
    if (!buffer) return NULL;
    
    char* text = (char*)malloc(buffer->length + 1);
    if (!text) return NULL;
    
    memcpy(text, buffer->data, buffer->gap_start);
    
    int text_after_gap = buffer->capacity - buffer->gap_end;
    memcpy(text + buffer->gap_start, buffer->data + buffer->gap_end, text_after_gap);
    
    text[buffer->length] = '\0';
    return text;
}

int text_buffer_get_length(TextBuffer* buffer) {
    if (!buffer) return 0;
    return buffer->length;
}