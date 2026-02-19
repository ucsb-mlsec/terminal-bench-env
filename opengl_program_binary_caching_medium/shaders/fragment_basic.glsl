#version 330 core

// Input from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

// Output color
out vec4 FragColor;

// Uniforms
uniform vec4 baseColor;
uniform sampler2D texture1;
uniform bool useTexture;

void main()
{
    vec4 texColor = vec4(1.0);
    
    if (useTexture) {
        texColor = texture(texture1, TexCoord);
    }
    
    // Simple ambient lighting
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    
    // Combine base color with texture and ambient
    FragColor = vec4(ambient, 1.0) * baseColor * texColor;
}