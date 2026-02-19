#version 450

#extension GL_NV_conservative_raster : enable

in vec3 fragPosition;
in vec3 fragNormal;
in vec2 fragTexCoord;

out vec4 fragColor;

uniform vec3 lightPosition;
uniform vec3 cameraPosition;

void main()
{
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(lightPosition - fragPosition);
    vec3 viewDir = normalize(cameraPosition - fragPosition);
    
    float diffuse = max(dot(normal, lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float specular = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    vec3 ambient = vec3(0.1, 0.1, 0.1);
    vec3 finalColor = ambient + diffuse * vec3(0.8, 0.8, 0.8) + specular * vec3(1.0, 1.0, 1.0);
    
    fragColor = vec4(finalColor, 1.0);
}