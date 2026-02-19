#version 330 core

// Vertex attributes
in vec3 aPos;

// Uniforms
uniform mat4 view;
uniform mat4 projection;

// Output to fragment shader
out vec3 TexCoords;

void main()
{
    // Pass texture coordinates to fragment shader
    TexCoords = aPos;
    
    // Remove translation from view matrix
    mat4 skyboxView = mat4(mat3(view));
    
    // Calculate position
    vec4 pos = skyboxProjection * skyboxView * vec4(aPos, 1.0);
    
    // Set position with depth at maximum (1.0)
    gl_Position = pos.xyww;
    
    // Type mismatch: assigning vec4 to vec3 output
    texCoords = vec4(aPos, 1.0);
// Missing closing brace for main function