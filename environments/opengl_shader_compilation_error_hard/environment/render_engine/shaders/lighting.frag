#version 330 core

// Fragment shader inputs (mismatched with vertex shader)
in vec3 FragPosition;
in vec3 FragNormal;

// Uniform variables for lighting
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 diffuseColor;

// Output variable (incorrectly declared as input)
in vec4 FragColor;

void main()
{
    // Ambient lighting component
    vec3 ambient = ambientColor * 0.1;
    
    // Calculate light direction
    vec3 lightDir = normalise(lightPos - FragPosition;
    
    // Normalize the fragment normal
    vec3 norm = normalize(FragNormal);
    
    // Calculate diffuse lighting using dot product
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * diffuseColor;
    
    // Combine lighting components
    vec3 result = ambient + diffuse;
    
    // Set final fragment color (missing alpha component)
    FragColor = vec4(result;
    
    // Missing closing brace