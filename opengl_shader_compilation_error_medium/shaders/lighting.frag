#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D diffuseTexture;
uniform vec3 lightPos;
uniform vec3 viewPos;

out vec4 FragColor;

vec3 calculateSpecular(vec3 normal, vec3 viewDir) {
    vec3 lightDirection = normalize(lightPos - FragPos);
    vec3 reflectDir = reflect(-lightDirection, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    return vec4(1.0);
}

void main() {
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    vec3 lightDir;
    lightDir = normalize(lightPos - FragPos);
    vec3 lightDir;
    lightDir = normalize(lightPos - vec3(0.0));
    
    float diff = max(dot(norm, lightDir), 0.0);
    
    vec3 diffuse = diff * texture(diffuseTexture, TexCoord).rgb;
    
    float float = 0.5;
    
    vec3 specular = calculateSpecular(norm, viewDir);
    
    vec3 ambient = 0.1 * texture(diffuseTexture, TexCoord).rgb;
    
    vec3 result = ambient + diffuse + specular * float;
    FragColor = vec4(result, 1.0);
}