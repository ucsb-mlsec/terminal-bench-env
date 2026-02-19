#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include "capture_system.h"

// Real-time video capture system using OpenGL
// Known performance issues reported: drops from 60 FPS to 15 FPS during capture

static GLuint pbo_id = 0;
static GLuint fbo_id = 0;
static int frame_width = 1920;
static int frame_height = 1080;
static unsigned char* temp_buffer = NULL;
static int capture_enabled = 0;

// Initialize the capture system with PBO for async readback
// Sets up pixel buffer object for GPU-CPU transfer
int init_capture_system(int width, int height) {
    frame_width = width;
    frame_height = height;
    
    // Allocate temporary CPU buffer for pixel data
    size_t buffer_size = width * height * 4; // RGBA
    temp_buffer = (unsigned char*)malloc(buffer_size);
    if (!temp_buffer) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        return -1;
    }
    
    // Generate and setup PBO for pixel readback
    glGenBuffers(1, &pbo_id);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id);
    glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, NULL, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    
    // Setup framebuffer for rendering
    glGenFramebuffers(1, &fbo_id);
    
    printf("Capture system initialized: %dx%d\n", width, height);
    return 0;
}

// Enable capture mode - activates frame grabbing
void enable_capture(int enable) {
    capture_enabled = enable;
    if (enable) {
        printf("Frame capture enabled\n");
    } else {
        printf("Frame capture disabled\n");
    }
}

// Bind the framebuffer for rendering target
void bind_capture_framebuffer(void) {
    if (fbo_id != 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
    }
}

// Main frame capture function - grabs current frame from GPU
// This is called every frame when capture is enabled
// PERFORMANCE ISSUE: Missing proper async mechanism
int capture_frame(void) {
    if (!capture_enabled) {
        return 0;
    }
    
    // Bind PBO as target for pixel read
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id);
    
    // Read pixels from current framebuffer into PBO
    // This should be async, but immediate mapping causes stall
    glReadPixels(0, 0, frame_width, frame_height, 
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);
    
    // CRITICAL ERROR: Immediately map buffer after read
    // This forces GPU-CPU sync and blocks pipeline!
    unsigned char* pixels = (unsigned char*)glMapBuffer(
        GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    
    if (pixels) {
        // Copy data to temp buffer for processing
        size_t data_size = frame_width * frame_height * 4;
        memcpy(temp_buffer, pixels, data_size);
        
        // Unmap the buffer
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    
    return 1;
}

// Direct pixel readback without PBO - extremely slow
// Used as fallback but causes severe stalls
void read_pixels_direct(int x, int y, int width, int height, 
                       unsigned char* output) {
    // CRITICAL BLOCKING OPERATION: Reading directly to CPU memory
    // This completely bypasses async mechanisms and forces immediate sync
    glReadPixels(x, y, width, height, 
                 GL_RGBA, GL_UNSIGNED_BYTE, output);
    
    // GPU must complete all rendering before this returns
    // Causes pipeline bubble and massive performance hit
}

// Process captured frame data for encoding/streaming
// Called after capture_frame to handle the pixel data
int process_captured_frame(void) {
    if (!temp_buffer || !capture_enabled) {
        return 0;
    }
    
    // Process the frame data in temp_buffer
    // In real application: compress, encode, stream, etc.
    
    // Simulate some processing
    size_t frame_size = frame_width * frame_height * 4;
    unsigned long checksum = 0;
    for (size_t i = 0; i < frame_size; i += 1024) {
        checksum += temp_buffer[i];
    }
    
    return 1;
}

// Get pointer to current frame buffer for external processing
unsigned char* get_frame_buffer(void) {
    return temp_buffer;
}

// Get current frame dimensions
void get_frame_dimensions(int* width, int* height) {
    if (width) *width = frame_width;
    if (height) *height = frame_height;
}

// Alternative capture using double mapping technique
// Attempts to optimize but still has issues
int capture_frame_optimized(void) {
    if (!capture_enabled) {
        return 0;
    }
    
    // Bind PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id);
    
    // Trigger async read into PBO
    glReadPixels(0, 0, frame_width, frame_height,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);
    
    // PROBLEM: Still no double buffering or proper rotation
    // Single PBO means we can't read new data while processing old
    // Missing fence sync or buffer rotation mechanism
    
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    
    return 1;
}

// Map the PBO to access captured data
// Separate mapping function for modular design
unsigned char* map_capture_buffer(void) {
    if (pbo_id == 0) {
        return NULL;
    }
    
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id);
    
    // Map buffer for reading
    // If called right after glReadPixels, this will block!
    unsigned char* data = (unsigned char*)glMapBuffer(
        GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    
    return data;
}

// Unmap the PBO after accessing data
void unmap_capture_buffer(void) {
    if (pbo_id == 0) {
        return;
    }
    
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

// Screenshot function for single frame capture
// High quality but blocking operation
int capture_screenshot(const char* filename) {
    size_t buffer_size = frame_width * frame_height * 4;
    unsigned char* screenshot_data = (unsigned char*)malloc(buffer_size);
    
    if (!screenshot_data) {
        return -1;
    }
    
    // Direct blocking read for screenshot
    // Acceptable here since it's one-time operation
    read_pixels_direct(0, 0, frame_width, frame_height, screenshot_data);
    
    // Save to file (simplified)
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(screenshot_data, 1, buffer_size, fp);
        fclose(fp);
    }
    
    free(screenshot_data);
    return 0;
}

// Cleanup and release resources
void cleanup_capture_system(void) {
    if (pbo_id != 0) {
        glDeleteBuffers(1, &pbo_id);
        pbo_id = 0;
    }
    
    if (fbo_id != 0) {
        glDeleteFramebuffers(1, &fbo_id);
        fbo_id = 0;
    }
    
    if (temp_buffer) {
        free(temp_buffer);
        temp_buffer = NULL;
    }
    
    capture_enabled = 0;
    printf("Capture system cleaned up\n");
}

// Query if capture is currently active
int is_capture_active(void) {
    return capture_enabled;
}

// Resize capture buffers for resolution changes
int resize_capture_buffers(int new_width, int new_height) {
    cleanup_capture_system();
    return init_capture_system(new_width, new_height);
}