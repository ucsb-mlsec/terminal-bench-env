#version 330 core

// Input vertex attributes
in vec3 aPosition;
in vec2 aTexCoord;

// Output to fragment shader
out vec2 vTexCoord;

void main()
{
    // Pass through texture coordinates
    vTexCoord = aTexCoord;
    
    // Transform vertex position to clip space
    // For post-processing, we render a full-screen quad
    // so we just pass through the position
    gl_Position = vec4(aPosition, 1.0);
}