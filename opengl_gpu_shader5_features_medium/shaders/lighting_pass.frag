#version 400
#extension GL_ARB_gpu_shader5 : enable

// Second pass for multi-pass lighting

uniform sampler2D gBufferPos;
uniform sampler2D gBufferNormal;
uniform sampler2D gBufferAlbedo;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

in vec2 vTexCoord;

out vec4 fragColor;

void main()
{
    vec3 fragPos = texture(gBufferPos, vTexCoord).rgb;
    vec3 normal = texture(gBufferNormal, vTexCoord).rgb;
    vec3 albedo = texture(gBufferAlbedo, vTexCoord).rgb;
    
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    
    vec3 result = (diffuse + specular) * albedo;
    fragColor = vec4(result, 1.0);
}