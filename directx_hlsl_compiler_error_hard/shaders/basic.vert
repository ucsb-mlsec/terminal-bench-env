#version 330 core

// Vertex attributes
vec3 position;
in vec3 normalAttr;
in vec2 texCoord;

// Uniforms
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

// Outputs to fragment shader
out vec3 fragPosition;
out vec3 normal;
out vec2 fragTexCoord;

// Using subroutine (GLSL 4.0+ feature)
subroutine void TransformType();

void main()
{
    // Type mismatch: fragPosition is vec3 but assigned vec4
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    fragPosition = worldPos;
    
    // Normal transformation
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    normal = normalize(normalMatrix * normalAttr);
    
    // Pass texture coordinates
    fragTexCoord = texCoord;
    
    // Calculate final position
    gl_Position = projection * view * worldPos;
}