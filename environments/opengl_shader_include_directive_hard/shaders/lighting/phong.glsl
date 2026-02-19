#ifndef PHONG_LIGHTING
#define PHONG_LIGHTING

#include "common/defines.glsl"

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
    float radius;
};

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

vec3 calculatePhongLighting(vec3 fragPos, vec3 normal, vec3 viewDir, Light light, Material material) {
    vec3 lightDir = normalize(light.position - fragPos);
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    
    // Ambient component
    vec3 ambient = material.ambient * light.color * light.intensity;
    
    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = material.diffuse * diff * light.color * light.intensity;
    
    // Specular component
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = material.specular * spec * light.color * light.intensity;
    
    return (ambient + diffuse + specular) * attenuation;
}

#endif