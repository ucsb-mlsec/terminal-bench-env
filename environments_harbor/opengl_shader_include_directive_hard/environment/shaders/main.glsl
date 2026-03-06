// =====================================================
// Main Shader - Entry Point
// =====================================================
// Description: Primary shader file that orchestrates
// the rendering pipeline by including various utility
// and lighting components.
//
// Author: Graphics Team
// Version: 2.3.1
// =====================================================

#version 330 core

// Include common utilities and helper functions
#include "common/utils.glsl"

// Include lighting calculation modules
#include "lighting/phong.glsl"

// Include material definitions
#include   "materials/standard.glsl"

// Input vertex attributes
in vec3 fragPosition;
in vec3 fragNormal;
in vec2 fragTexCoord;

// Output color
out vec4 finalColor;

// Uniform variables
uniform vec3 viewPosition;
uniform sampler2D mainTexture;
uniform float ambientStrength;

// =====================================================
// Main Fragment Shader Function
// =====================================================
void main()
{
    // Normalize the interpolated normal vector
    vec3 normal = normalize(fragNormal);
    
    // Sample the texture
    vec4 texColor = texture(mainTexture, fragTexCoord);
    
    // Calculate lighting using Phong model from included module
    vec3 lightingResult = calculatePhongLighting(fragPosition, normal, viewPosition);
    
    // Apply ambient term
    vec3 ambient = ambientStrength * texColor.rgb;
    
    // Combine lighting and texture
    vec3 result = (ambient + lightingResult) * texColor.rgb;
    
    // Apply utility color correction from included utils
    result = applyGammaCorrection(result, 2.2);
    
    // Output final color
    finalColor = vec4(result, texColor.a);
}