#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec3 fragPos;
in vec3 normal;
in vec4 fragPosLightSpace;

out vec4 fragColor;

uniform sampler2D shadowMap;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float shadowBias;

void main()
{
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    vec3 viewDir = normalize(viewPos - fragPos);
    
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    // PCF (Percentage Closer Filtering) for soft shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    float currentDepth = projCoords.z;
    
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            vec2 offset = vec2(x, y) * texelSize;
            float pcfDepth = texture(shadowMap, projCoords.xy + offset).r;
            float shadowValue = currentDepth - shadowBias > pcfDepth ? 1.0 : 0.0;
            float sampleWeight = 1.0 / 9.0;
            shadow = fma(sampleWeight, shadowValue, shadow);
        }
    }
    
    // Calculate diffuse lighting
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = vec3(diff);
    
    // Calculate specular lighting
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 32.0);
    vec3 specular = vec3(spec) * 0.5;
    
    // Combine lighting with shadow
    vec3 ambient = vec3(0.15);
    vec3 lighting = fma(diffuse, vec3(1.0 - shadow), ambient);
    lighting = fma(specular, vec3(1.0 - shadow), lighting);
    
    fragColor = vec4(lighting, 1.0);
}