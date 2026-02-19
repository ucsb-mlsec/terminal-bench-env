#version 330 core

// Include directives with issues
#include "/includes/color_utils.glsl"
#include common/math_helpers.glsl"
#include "/common/constants.glsl"

in vec3 fragNormal;
in vec3 fragPos;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    // Normalize the normal vector
    vec3 normal = normalize(fragNormal);
    
    // Calculate light direction
    vec3 lightDir = normalize(lightPos - fragPos);
    
    // Calculate view direction
    vec3 viewDir = normalize(viewPos - fragPos);
    
    // Use function from color_utils.glsl (should exist)
    vec3 baseColor = vec3(0.8, 0.6, 0.4);
    
    // Use function from math_helpers.glsl
    float lighting = calculateLighting(normal, lightDir, viewDir);
    
    // Apply gamma correction from color_utils
    vec3 finalColor = applyGamma(baseColor * lighting);
    
    FragColor = vec4(finalColor, 1.0);
}