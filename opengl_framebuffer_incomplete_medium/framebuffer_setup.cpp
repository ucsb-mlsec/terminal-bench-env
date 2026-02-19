#include <GL/gl.h>
#include <iostream>

// Framebuffer initialization routine for offscreen rendering
// Target resolution: 1024x768
// This code is extracted from the main graphics application for debugging

GLuint setupOffscreenFramebuffer() {
    GLuint framebuffer;
    GLuint colorTexture;
    GLuint depthRenderbuffer;
    
    const int TARGET_WIDTH = 1024;
    const int TARGET_HEIGHT = 768;
    
    std::cout << "Initializing offscreen framebuffer at " << TARGET_WIDTH << "x" << TARGET_HEIGHT << std::endl;
    
    // Generate and bind the framebuffer object
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    
    // ===== COLOR ATTACHMENT SETUP =====
    // Create color texture for rendering
    glGenTextures(1, &colorTexture);
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Allocate texture storage for color attachment
    // NOTE: This should match our target resolution
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 0, TARGET_HEIGHT, 0, 
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    
    // Attach color texture to framebuffer
    // Using attachment point for additional render targets
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 
                          GL_TEXTURE_2D, colorTexture, 0);
    
    std::cout << "Color texture attached" << std::endl;
    
    // ===== DEPTH ATTACHMENT SETUP =====
    // Create depth renderbuffer for depth testing
    glGenRenderbuffers(1, &depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
    
    // Configure depth renderbuffer storage
    // Using a standard depth renderbuffer size
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
    
    // Attach depth renderbuffer to framebuffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
                             GL_RENDERBUFFER, depthRenderbuffer);
    
    std::cout << "Depth renderbuffer attached" << std::endl;
    
    // ===== ADDITIONAL STENCIL SETUP =====
    // Adding a stencil attachment for advanced rendering effects
    GLuint stencilRenderbuffer;
    glGenRenderbuffers(1, &stencilRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, stencilRenderbuffer);
    
    // Note: Allocate storage for stencil buffer
    // Using GL_STENCIL_INDEX format for stencil operations
    glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX, TARGET_WIDTH, TARGET_HEIGHT);
    
    // Attach stencil renderbuffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                             GL_RENDERBUFFER, stencilRenderbuffer);
    
    std::cout << "Stencil renderbuffer attached" << std::endl;
    
    // ===== FRAMEBUFFER STATUS CHECK =====
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR: Framebuffer is incomplete!" << std::endl;
        std::cerr << "Status code: 0x" << std::hex << status << std::endl;
        
        // Print common causes
        if (status == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
            std::cerr << "Cause: INCOMPLETE_ATTACHMENT" << std::endl;
        } else if (status == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT) {
            std::cerr << "Cause: MISSING_ATTACHMENT" << std::endl;
        } else if (status == GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT) {
            std::cerr << "Cause: INCOMPLETE_DIMENSIONS" << std::endl;
        } else if (status == GL_FRAMEBUFFER_UNSUPPORTED) {
            std::cerr << "Cause: UNSUPPORTED format combination" << std::endl;
        }
        
        // Cleanup on failure
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteTextures(1, &colorTexture);
        glDeleteRenderbuffers(1, &depthRenderbuffer);
        glDeleteRenderbuffers(1, &stencilRenderbuffer);
        
        return 0;
    }
    
    std::cout << "Framebuffer setup complete and validated" << std::endl;
    
    // Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    return framebuffer;
}