#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

void main()
{
    // Sample G-buffer textures
    vec3 fragPos = texture(gPosition, texCoord).rgb;
    vec3 normal = normalize(texture(gNormal, texCoord).rgb);
    vec3 albedo = texture(gAlbedo, texCoord).rgb;
    
    // Initialize lighting accumulator
    vec3 lighting = vec3(0.0);
    
    // Calculate lighting contribution from each light
    for(int i = 0; i < 4; i++)
    {
        vec3 lightDir = lightPositions[i] - fragPos;
        float distance = length(lightDir);
        lightDir = normalize(lightDir);
        
        // Calculate attenuation
        float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
        
        // Diffuse lighting
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = diff * lightColors[i] * attenuation;
        
        // Accumulate lighting using FMA
        vec3 lightContribution = diffuse;
        lighting = fma(lightContribution, lightColors[i], lighting);
    }
    
    // Add ambient lighting
    vec3 ambient = vec3(0.03) * albedo;
    
    // Combine lighting with albedo using FMA
    vec3 final = fma(lighting, albedo, ambient);
    
    fragColor = vec4(final, 1.0);
}