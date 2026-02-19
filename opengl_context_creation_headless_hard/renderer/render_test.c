#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <EGL/egl.h>
#include <GL/gl.h>

#define WIDTH 64
#define HEIGHT 64
#define OUTPUT_PATH "output/test.ppm"

// Function to check EGL errors
void checkEGLError(const char* msg) {
    EGLint error = eglGetError();
    if (error != EGL_SUCCESS) {
        fprintf(stderr, "EGL Error at %s: 0x%x\n", msg, error);
        exit(1);
    }
}

// Function to check OpenGL errors
void checkGLError(const char* msg) {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL Error at %s: 0x%x\n", msg, error);
        exit(1);
    }
}

int main() {
    EGLDisplay display;
    EGLConfig config;
    EGLContext context;
    EGLSurface surface;
    EGLint numConfigs;
    EGLint major, minor;

    // Bug 1: Not using EGL_DEFAULT_DISPLAY for headless rendering
    // Should use EGL_DEFAULT_DISPLAY or proper platform display
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display\n");
        return 1;
    }

    if (!eglInitialize(display, &major, &minor)) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return 1;
    }
    checkEGLError("eglInitialize");

    printf("EGL Version: %d.%d\n", major, minor);

    // Bug 2: Incorrect EGL configuration attributes for pbuffer
    // Missing EGL_PBUFFER_BIT in EGL_SURFACE_TYPE
    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,  // Bug: should be EGL_PBUFFER_BIT
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE
    };

    if (!eglChooseConfig(display, configAttribs, &config, 1, &numConfigs)) {
        fprintf(stderr, "Failed to choose EGL config\n");
        eglTerminate(display);
        return 1;
    }
    checkEGLError("eglChooseConfig");

    if (numConfigs == 0) {
        fprintf(stderr, "No matching EGL configs found\n");
        eglTerminate(display);
        return 1;
    }

    // Bug 3: Need to bind OpenGL API
    // Missing eglBindAPI(EGL_OPENGL_API)

    // Create pbuffer surface
    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, WIDTH,
        EGL_HEIGHT, HEIGHT,
        EGL_NONE
    };

    surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
    if (surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Failed to create pbuffer surface\n");
        eglTerminate(display);
        return 1;
    }
    checkEGLError("eglCreatePbufferSurface");

    // Create OpenGL context
    const EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 2,
        EGL_CONTEXT_MINOR_VERSION, 1,
        EGL_NONE
    };

    context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create EGL context\n");
        eglDestroySurface(display, surface);
        eglTerminate(display);
        return 1;
    }
    checkEGLError("eglCreateContext");

    // Make context current
    if (!eglMakeCurrent(display, surface, surface, context)) {
        fprintf(stderr, "Failed to make context current\n");
        eglDestroyContext(display, context);
        eglDestroySurface(display, surface);
        eglTerminate(display);
        return 1;
    }
    checkEGLError("eglMakeCurrent");

    // Set up OpenGL viewport
    glViewport(0, 0, WIDTH, HEIGHT);
    checkGLError("glViewport");

    // Bug 4: Wrong clear color - using blue instead of red
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);  // Bug: should be (1.0f, 0.0f, 0.0f, 1.0f)
    checkGLError("glClearColor");

    glClear(GL_COLOR_BUFFER_BIT);
    checkGLError("glClear");

    // Ensure rendering is complete
    glFinish();
    checkGLError("glFinish");

    // Read pixels from framebuffer
    unsigned char* pixels = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
    if (!pixels) {
        fprintf(stderr, "Failed to allocate pixel buffer\n");
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(display, context);
        eglDestroySurface(display, surface);
        eglTerminate(display);
        return 1;
    }

    // Bug 5: Wrong pixel format - using GL_BGR instead of GL_RGB
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, pixels);
    checkGLError("glReadPixels");

    // Write PPM file
    FILE* fp = fopen(OUTPUT_PATH, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file: %s\n", OUTPUT_PATH);
        free(pixels);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(display, context);
        eglDestroySurface(display, surface);
        eglTerminate(display);
        return 1;
    }

    // Write PPM header
    fprintf(fp, "P3\n%d %d\n255\n", WIDTH, HEIGHT);

    // Write pixel data (flip vertically because OpenGL origin is bottom-left)
    for (int y = HEIGHT - 1; y >= 0; y--) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = (y * WIDTH + x) * 3;
            fprintf(fp, "%d %d %d\n", 
                    pixels[idx], 
                    pixels[idx + 1], 
                    pixels[idx + 2]);
        }
    }

    fclose(fp);
    free(pixels);

    printf("Successfully rendered to %s\n", OUTPUT_PATH);

    // Cleanup
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context);
    eglDestroySurface(display, surface);
    eglTerminate(display);

    return 0;
}