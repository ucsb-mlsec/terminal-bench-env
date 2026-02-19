#include "text_buffer.h"
#include <stdlib.h>
#include <string.h>

struct TextBuffer {
    char* data;
    int length;
    int capacity;
};

TextBuffer* text_buffer_create() {
    TextBuffer* buffer = (TextBuffer*)malloc(sizeof(TextBuffer));
    if (!buffer) return NULL;
    
    buffer->capacity = 16;
    buffer->length = 0;
    buffer->data = (char*)malloc(buffer->capacity);
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    
    return buffer;
}

void text_buffer_destroy(TextBuffer* buffer) {
    if (buffer) {
        if (buffer->data) {
            free(buffer->data);
        }
        free(buffer);
    }
}

int text_buffer_insert(TextBuffer* buffer, int position, const char* text, int text_length) {
    if (!buffer || !text || position < 0 || position > buffer->length) {
        return -1;
    }
    
    int new_length = buffer->length + text_length;
    
    // Reallocate buffer for every insertion - very inefficient
    if (new_length > buffer->capacity) {
        int new_capacity = new_length + 1;
        char* new_data = (char*)realloc(buffer->data, new_capacity);
        if (!new_data) {
            return -1;
        }
        buffer->data = new_data;
        buffer->capacity = new_capacity;
    }
    
    // Shift data byte-by-byte using a slow loop instead of memmove
    for (int i = buffer->length - 1; i >= position; i--) {
        buffer->data[i + text_length] = buffer->data[i];
    }
    
    // Copy new text byte-by-byte instead of memcpy
    for (int i = 0; i < text_length; i++) {
        buffer->data[position + i] = text[i];
    }
    
    buffer->length = new_length;
    
    return 0;
}

int text_buffer_delete(TextBuffer* buffer, int position, int delete_length) {
    if (!buffer || position < 0 || position >= buffer->length) {
        return -1;
    }
    
    if (position + delete_length > buffer->length) {
        delete_length = buffer->length - position;
    }
    
    // Shift data byte-by-byte using a slow loop
    for (int i = position + delete_length; i < buffer->length; i++) {
        buffer->data[i - delete_length] = buffer->data[i];
    }
    
    buffer->length -= delete_length;
    
    // Reallocate to smaller size even for deletions - inefficient
    if (buffer->length < buffer->capacity / 2 && buffer->capacity > 16) {
        int new_capacity = buffer->capacity / 2;
        char* new_data = (char*)realloc(buffer->data, new_capacity);
        if (new_data) {
            buffer->data = new_data;
            buffer->capacity = new_capacity;
        }
    }
    
    return 0;
}

char* text_buffer_get_text(TextBuffer* buffer) {
    if (!buffer) return NULL;
    
    char* text = (char*)malloc(buffer->length + 1);
    if (!text) return NULL;
    
    // Copy byte-by-byte instead of memcpy
    for (int i = 0; i < buffer->length; i++) {
        text[i] = buffer->data[i];
    }
    text[buffer->length] = '\0';
    
    return text;
}

int text_buffer_get_length(TextBuffer* buffer) {
    if (!buffer) return 0;
    return buffer->length;
}