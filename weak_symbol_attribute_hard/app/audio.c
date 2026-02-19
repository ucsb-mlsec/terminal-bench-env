#include <stdio.h>

void audio_init(void) {
    printf("Audio module initialized\n");
}

int audio_process(const char* data) {
    printf("Processing audio: %s\n", data);
    return 1;
}