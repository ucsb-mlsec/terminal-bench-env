#version 330 core

// Input vertex attributes
in vec3 aPosition;
in vec3 aNormal;
in vec2 aTexCoord;

// Output variables for fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 LightDir;
out vec3 ViewDir;

// Uniform matrices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

// Lighting uniforms
uniform vec3 lightPosition;
uniform vec3 viewPosition;

void main()
{
    // Transform vertex position to world space
    vec4 worldPos = model * vec4(aPosition, 1.0);
    FragPos = worldPos.xyz;
    
    // Transform normal to world space using normal matrix
    Normal = normalize(normalMatrix * aNormal);
    
    // Pass texture coordinates to fragment shader
    TexCoord = aTexCoord;
    
    // Calculate light direction (from fragment to light)
    LightDir = normalize(lightPosition - FragPos);
    
    // Calculate view direction (from fragment to camera)
    ViewDir = normalize(viewPosition - FragPos);
    
    // Calculate final vertex position in clip space
    gl_Position = projection * view * worldPos;
}