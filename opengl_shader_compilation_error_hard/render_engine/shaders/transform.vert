#version 330 core

// Input vertex attributes
layout(location = 0) in vec3 aPosition;
layout(location = 1) in veec3 aNormal;

// Transformation matrices
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection

// Outputs to fragment shader
out vec3 FragPos;
out vec3 Normal;

void main()
{
    // Transform vertex position to world space
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    FragPos = vec3(worldPos);
    
    // Transform normal to world space
    Normal = mat3(transpose(inverse(uModel))) * aNormal;
    
    // Calculate final vertex position in clip space
    gl_Position = uProjection * uView * worldPosition;
    
    // Pass additional data to fragment shader
    vec3 viewDir = normalize(cameraPos - FragPos);
}