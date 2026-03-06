#ifndef TEXT_BUFFER_H
#define TEXT_BUFFER_H

#include <stdlib.h>
#include <string.h>

typedef struct TextBuffer TextBuffer;

TextBuffer* text_buffer_create(void);
void text_buffer_destroy(TextBuffer* buffer);
void text_buffer_insert(TextBuffer* buffer, int position, const char* text);
void text_buffer_delete(TextBuffer* buffer, int position, int length);
char* text_buffer_get_text(TextBuffer* buffer);
int text_buffer_get_length(TextBuffer* buffer);

#endif