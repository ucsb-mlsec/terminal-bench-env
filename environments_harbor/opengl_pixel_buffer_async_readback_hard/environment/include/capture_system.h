#ifndef CAPTURE_SYSTEM_H
#define CAPTURE_SYSTEM_H

#include <GL/gl.h>
#include <GL/glext.h>

#define NUM_PBOS 2
#define CAPTURE_FORMAT GL_RGBA
#define CAPTURE_TYPE GL_UNSIGNED_BYTE

typedef struct {
    int width;
    int height;
    GLuint pbos[NUM_PBOS];
    int current_pbo;
    int initialized;
} CaptureContext;

/* Initialize the capture system with given dimensions */
int init_capture_system(CaptureContext* ctx, int width, int height);

/* Capture the current frame from the framebuffer */
int capture_frame(CaptureContext* ctx);

/* Read captured frame data into the provided buffer */
int read_frame_data(CaptureContext* ctx, unsigned char* buffer);

/* Get the size of a frame in bytes */
int get_frame_size(CaptureContext* ctx);

/* Cleanup and release all resources */
void cleanup_capture(CaptureContext* ctx);

/* Check if capture system is ready */
int is_capture_ready(CaptureContext* ctx);

#endif /* CAPTURE_SYSTEM_H */