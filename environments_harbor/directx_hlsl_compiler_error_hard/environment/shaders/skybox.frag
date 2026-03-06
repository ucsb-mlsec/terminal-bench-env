#version 330 core

// Missing input declaration for texCoords
// Should have: in vec3 texCoords;

// Wrong sampler type - should be samplerCube, not sampler2D
uniform sampler2D skybox;

// Missing output color declaration
// Should have: out vec4 FragColor;

void main()
{
    // Syntax error: missing comma between arguments
    // Should be: texture(skybox, texCoords)
    vec4 color = texture(skybox texCoords);
    
    // This line will fail because FragColor is not declared
    FragColor = color;
}