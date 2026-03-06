#include <GL/glew.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>

#include "renderer.h"
#include "shader.h"
#include "texture.h"

// Global renderer state
static int viewport_width = 1920;
static int viewport_height = 1080;
static float clear_color[4] = {0.1f, 0.1f, 0.15f, 1.0f};

/**
 * Check for OpenGL errors and print error messages
 */
static void check_gl_error(const char* operation) {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error after %s: 0x%x\n", operation, error);
    }
}

/**
 * Initialize the renderer and set up OpenGL state
 * This function must be called after the OpenGL context is created
 */
int init_renderer(void) {
    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        return 0;
    }
    
    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    
    // Set clear color for the background
    glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
    check_gl_error("glClearColor");
    
    // Enable depth testing for proper 3D rendering
    glEnable(GL_DEPTH_TEST);
    check_gl_error("glEnable(GL_DEPTH_TEST)");
    
    // Set depth test function to less than or equal
    glDepthFunc(GL_LEQUAL);
    check_gl_error("glDepthFunc");
    
    // Enable back-face culling for performance
    glEnable(GL_CULL_FACE);
    check_gl_error("glEnable(GL_CULL_FACE)");
    
    // Cull back faces
    glCullFace(GL_BACK);
    check_gl_error("glCullFace");
    
    // Set front face winding order to counter-clockwise
    glFrontFace(GL_CCW);
    check_gl_error("glFrontFace");
    
    // Enable alpha blending for transparency
    glEnable(GL_BLEND);
    check_gl_error("glEnable(GL_BLEND)");
    
    // Set standard alpha blending function
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    check_gl_error("glBlendFunc");
    
    // Enable multisampling for anti-aliasing if available
    glEnable(GL_MULTISAMPLE);
    check_gl_error("glEnable(GL_MULTISAMPLE)");
    
    // Enable cube map textures
    glEnable(GL_TEXTURE_CUBE_MAP);
    check_gl_error("glEnable(GL_TEXTURE_CUBE_MAP)");
    
    // Enable program point size for point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    check_gl_error("glEnable(GL_PROGRAM_POINT_SIZE)");
    
    // Enable primitive restart for indexed rendering optimization
    glEnable(GL_PRIMITIVE_RESTART);
    check_gl_error("glEnable(GL_PRIMITIVE_RESTART)");
    
    // Set primitive restart index
    glPrimitiveRestartIndex(0xFFFFFFFF);
    check_gl_error("glPrimitiveRestartIndex");
    
    // Enable line smoothing for better quality lines
    glEnable(GL_LINE_SMOOTH);
    check_gl_error("glEnable(GL_LINE_SMOOTH)");
    
    // Set line smoothing hint to nicest quality
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    check_gl_error("glHint(GL_LINE_SMOOTH_HINT)");
    
    // Enable polygon offset for depth fighting prevention
    glEnable(GL_POLYGON_OFFSET_FILL);
    check_gl_error("glEnable(GL_POLYGON_OFFSET_FILL)");
    
    // Set polygon offset parameters
    glPolygonOffset(1.0f, 1.0f);
    check_gl_error("glPolygonOffset");
    
    // Disable dithering for consistent colors
    glDisable(GL_DITHER);
    check_gl_error("glDisable(GL_DITHER)");
    
    // Set pixel storage alignment
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    check_gl_error("glPixelStorei");
    
    printf("Renderer initialized successfully\n");
    return 1;
}

/**
 * Setup viewport and projection parameters
 */
void setup_viewport(int width, int height) {
    viewport_width = width;
    viewport_height = height;
    
    // Set the viewport to cover the entire window
    glViewport(0, 0, width, height);
    check_gl_error("glViewport");
    
    printf("Viewport configured: %dx%d\n", width, height);
}

/**
 * Update clear color
 */
void set_clear_color(float r, float g, float b, float a) {
    clear_color[0] = r;
    clear_color[1] = g;
    clear_color[2] = b;
    clear_color[3] = a;
    
    glClearColor(r, g, b, a);
    check_gl_error("glClearColor");
}

/**
 * Clear the framebuffer
 */
void clear_buffers(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    check_gl_error("glClear");
}

/**
 * Get viewport dimensions
 */
void get_viewport_size(int* width, int* height) {
    if (width) *width = viewport_width;
    if (height) *height = viewport_height;
}

/**
 * Enable wireframe rendering mode
 */
void set_wireframe_mode(int enabled) {
    if (enabled) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    check_gl_error("glPolygonMode");
}

/**
 * Cleanup renderer resources
 */
void cleanup_renderer(void) {
    // Disable all OpenGL capabilities that were enabled
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glDisable(GL_MULTISAMPLE);
    glDisable(GL_TEXTURE_CUBE_MAP);
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_PRIMITIVE_RESTART);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_OFFSET_FILL);
    
    printf("Renderer cleaned up\n");
}