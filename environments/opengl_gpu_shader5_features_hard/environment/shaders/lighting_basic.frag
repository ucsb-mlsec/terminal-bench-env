#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec3 normal;
in vec2 texCoord;
in vec3 fragPos;

out vec4 fragColor;

uniform vec3 lightColor;
uniform vec3 lightPosition;
uniform float intensity;
uniform vec3 viewPos;
uniform vec3 objectColor;

void main()
{
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPosition - fragPos);
    
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = lightColor * diff;
    diffuse = diffuse * intensity;
    
    vec3 ambient = lightColor * 0.1;
    
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = lightColor * spec;
    
    vec3 result = ambient + diffuse;
    result = result + specular;
    result = result * objectColor;
    
    fragColor = vec4(result, 1.0);
}