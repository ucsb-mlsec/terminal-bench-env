#version 330 core

// Input vertex attributes
in vec3 position;
in vec3 normal;
in vec2 texCoord;

// Output variables to fragment shader
out vec3 fragNormal;
out vec2 fragTexCoord;
out vec3 fragPosition;

// Uniform matrices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Transform vertex position to clip space
    mat4 mvp = projection * view * model;
    gl_Position = mvp * vec4(position, 1.0);
    
    // Pass world space position to fragment shader
    fragPosition = vec3(model * vec4(position, 1.0));
    
    // Transform normal to world space
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    fragNormal = normalize(normalMatrix * normal);
    
    // Pass texture coordinates to fragment shader
    fragTexCoord = texCoord;
}