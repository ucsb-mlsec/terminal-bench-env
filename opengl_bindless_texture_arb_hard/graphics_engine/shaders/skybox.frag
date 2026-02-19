#version 450 core

// Skybox Fragment Shader
// Renders a cubemap environment for background scenery
// Uses traditional texture sampling with samplerCube

// Input texture coordinate from vertex shader
in vec3 texCoord;

// Final output color
out vec4 fragColor;

// Cubemap texture sampler - traditional binding method
uniform samplerCube skyboxTexture;

// Gamma correction factor
const float gamma = 2.2;

void main()
{
    // Sample the cubemap texture using the interpolated direction vector
    vec3 envColor = texture(skyboxTexture, texCoord).rgb;
    
    // Apply gamma correction for proper color space handling
    // Convert from linear to sRGB space
    envColor = pow(envColor, vec3(1.0 / gamma));
    
    // Output the final skybox color with full opacity
    // Skybox is always at maximum depth, so no blending needed
    fragColor = vec4(envColor, 1.0);
}