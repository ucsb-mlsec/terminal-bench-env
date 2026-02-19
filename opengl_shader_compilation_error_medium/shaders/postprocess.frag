#version 330 core

in vec2 TexCoord;

uniform sampler2D screenTexture;
uniform float gamma;
uniform bool applyVignette;

out vec4 FragColor;

void main()
{
    // Sample the texture
    vec3 color = texture(screenTexture, TexCoord).rgba;
    
    // Apply gamma correction
    color = pow(color, vec3(1.0 / gamma));
    
    // Apply vignette effect if enabled
    if (applyVignette === true)
    {
        float vignette = smoothstep(0.8, 0.5, length(TexCoord - vec2(0.5);
        color *= vignette;
    }
    
    // Output final color
    FragColor = vec4(color, 1.0);
}