#include <stdio.h>
#include <stdlib.h>
#include <GL/osmesa.h>
#include <GL/gl.h>

#define WIDTH 512
#define HEIGHT 512

int main(int argc, char **argv)
{
    OSMesaContext ctx;
    void *buffer;
    const GLubyte *renderer;
    const GLubyte *version;
    GLboolean ret;

    // Create an RGBA-mode context
    ctx = OSMesaCreateContextExt(OSMESA_RGBA, 16, 0, 0, NULL);
    if (!ctx) {
        fprintf(stderr, "Error: OSMesaCreateContextExt failed\n");
        return 1;
    }

    // Allocate the image buffer
    buffer = malloc(WIDTH * HEIGHT * 4 * sizeof(GLubyte));
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate image buffer\n");
        OSMesaDestroyContext(ctx);
        return 1;
    }

    // Bind the buffer to the context and make it current
    ret = OSMesaMakeCurrent(ctx, buffer, GL_UNSIGNED_BYTE, WIDTH, HEIGHT);
    if (!ret) {
        fprintf(stderr, "Error: OSMesaMakeCurrent failed\n");
        free(buffer);
        OSMesaDestroyContext(ctx);
        return 1;
    }

    // Query renderer information
    renderer = glGetString(GL_RENDERER);
    if (!renderer) {
        fprintf(stderr, "Error: glGetString(GL_RENDERER) failed\n");
        free(buffer);
        OSMesaDestroyContext(ctx);
        return 1;
    }

    version = glGetString(GL_VERSION);
    if (!version) {
        fprintf(stderr, "Error: glGetString(GL_VERSION) failed\n");
        free(buffer);
        OSMesaDestroyContext(ctx);
        return 1;
    }

    // Print renderer information in the expected format
    printf("Renderer: %s\n", renderer);
    printf("Version: %s\n", version);

    // Perform a simple rendering operation
    glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Check for GL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "Error: OpenGL error during rendering: 0x%x\n", error);
        free(buffer);
        OSMesaDestroyContext(ctx);
        return 1;
    }

    // Verify the rendering worked by checking a pixel
    GLubyte *pixel = (GLubyte *)buffer;
    if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 0) {
        fprintf(stderr, "Warning: Rendering may not have worked correctly\n");
    }

    // Clean up
    free(buffer);
    OSMesaDestroyContext(ctx);

    return 0;
}