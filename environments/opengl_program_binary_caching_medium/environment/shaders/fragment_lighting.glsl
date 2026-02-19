#version 330 core

// Input from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

// Output color
out vec4 FragColor;

// Material properties
uniform vec3 materialAmbient;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;
uniform float materialShininess;

// Light properties
uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform float lightIntensity;

// Camera position
uniform vec3 viewPosition;

void main()
{
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    
    // Calculate light direction
    vec3 lightDir = normalize(lightPosition - FragPos);
    
    // Ambient component
    vec3 ambient = materialAmbient * lightColor * 0.1;
    
    // Diffuse component
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = materialDiffuse * diff * lightColor * lightIntensity;
    
    // Specular component (Blinn-Phong)
    vec3 viewDir = normalize(viewPosition - FragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), materialShininess);
    vec3 specular = materialSpecular * spec * lightColor * lightIntensity;
    
    // Attenuation based on distance
    float distance = length(lightPosition - FragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    
    // Apply attenuation to lighting components
    diffuse *= attenuation;
    specular *= attenuation;
    
    // Combine all components
    vec3 result = ambient + diffuse + specular;
    
    // Output final color
    FragColor = vec4(result, 1.0);
}