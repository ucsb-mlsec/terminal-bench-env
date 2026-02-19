#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 1024

int save_data(const char *filename, const char *data, size_t size) {
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        return -1;
    }
    size_t written = fwrite(data, 1, size, fp);
    fclose(fp);
    return (written == size) ? 0 : -1;
}

int load_data(const char *filename, char *buffer, size_t buffer_size) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        return -1;
    }
    size_t bytes_read = fread(buffer, 1, buffer_size - 1, fp);
    buffer[bytes_read] = '\0';
    fclose(fp);
    return bytes_read;
}

int delete_record(const char *filename) {
    if (remove(filename) == 0) {
        return 0;
    }
    return -1;
}

int append_data(const char *filename, const char *data, size_t size) {
    FILE *fp = fopen(filename, "ab");
    if (fp == NULL) {
        return -1;
    }
    size_t written = fwrite(data, 1, size, fp);
    fclose(fp);
    return (written == size) ? 0 : -1;
}