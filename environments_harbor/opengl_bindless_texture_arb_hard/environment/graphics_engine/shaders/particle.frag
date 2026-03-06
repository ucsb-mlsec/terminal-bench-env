#version 450 core

// Particle rendering fragment shader
// Uses traditional texture sampling with uniform sampler2D

// Traditional sampler uniform - incompatible with bindless textures
uniform sampler2D particleTexture;

// Tint color for the particle system
uniform vec4 tintColor;

// Input from vertex shader
in vec2 texCoord;
in vec4 particleColor;

// Final fragment color output
out vec4 fragColor;

void main()
{
    // Sample the particle texture using traditional texture sampling
    vec4 texColor = texture(particleTexture, texCoord);
    
    // Combine texture color with particle color and tint
    vec4 finalColor = texColor * particleColor * tintColor;
    
    // Apply alpha blending
    // Premultiply alpha for correct blending
    finalColor.rgb *= finalColor.a;
    
    // Discard fully transparent fragments to improve performance
    if (finalColor.a < 0.01)
    {
        discard;
    }
    
    // Output final color
    fragColor = finalColor;
}