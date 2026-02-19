#version 450 core

// Input variables from vertex shader
in vec3 fragPosition;
in vec3 fragNormal;
in vec2 fragTexCoord;

// Output variable
out vec4 FragColor;

// Uniform variables
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 baseColor;
uniform float roughness;
uniform float metallic;

// Subroutine type declarations
subroutine vec3 MaterialShading();
subroutine vec4 MaterialColor();
subroutine float MaterialRoughness();

// Subroutine uniform declarations
subroutine uniform MaterialShading materialFunc;
subroutine uniform MaterialColor colorFunc;
subroutine MaterialRoughness roughnessFunc;  // ERROR: missing 'uniform' keyword

// Helper function for lighting calculations
vec3 calculateLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 albedo) {
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    return albedo * diff + vec3(0.3) * spec;
}

// Correct subroutine implementation for Phong shading
subroutine(MaterialShading)
vec3 phongShading() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    vec3 viewDir = normalize(viewPos - fragPosition);
    
    vec3 ambient = 0.1 * baseColor;
    vec3 diffuse = max(dot(normal, lightDir), 0.0) * baseColor;
    
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = 0.5 * spec * vec3(1.0);
    
    return ambient + diffuse + specular;
}

// ERROR: Missing 'subroutine' keyword before function definition
vec3 pbrMetallic() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    vec3 viewDir = normalize(viewPos - fragPosition);
    
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);
    vec3 halfDir = normalize(viewDir + lightDir);
    
    float NdotH = max(dot(normal, halfDir), 0.0);
    float roughnessSq = roughness * roughness;
    float D = roughnessSq / (3.14159 * pow(NdotH * NdotH * (roughnessSq - 1.0) + 1.0, 2.0));
    
    return baseColor * D * 0.5;
}

// ERROR: Wrong subroutine type association (should be MaterialShading, not MaterialColor)
subroutine(MaterialColor)
vec3 toonShading() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    
    float intensity = dot(normal, lightDir);
    vec3 color;
    
    if (intensity > 0.95)
        color = baseColor;
    else if (intensity > 0.5)
        color = baseColor * 0.7;
    else if (intensity > 0.25)
        color = baseColor * 0.4;
    else
        color = baseColor * 0.2;
    
    return color;
}

// Correct subroutine implementation
subroutine(MaterialShading)
vec3 diffuseOnly() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    float diff = max(dot(normal, lightDir), 0.0);
    return baseColor * diff;
}

// ERROR: Incorrect return type (returns vec4 instead of vec3 for MaterialShading)
subroutine(MaterialShading)
vec4 emissiveShading() {
    return vec4(baseColor * 2.0, 1.0);
}

// Correct subroutine implementation for MaterialColor
subroutine(MaterialColor)
vec4 solidColor() {
    return vec4(baseColor, 1.0);
}

// Correct subroutine implementation for MaterialColor
subroutine(MaterialColor)
vec4 texturedColor() {
    return vec4(baseColor * (1.0 + 0.2 * sin(fragTexCoord.x * 10.0)), 1.0);
}

// Main shader function
void main() {
    // Call the material shading subroutine
    vec3 shadedColor = materialFunc();
    
    // Call the color subroutine
    vec4 finalColor = colorFunc();
    
    // Combine results
    FragColor = vec4(shadedColor, 1.0) * finalColor;
}