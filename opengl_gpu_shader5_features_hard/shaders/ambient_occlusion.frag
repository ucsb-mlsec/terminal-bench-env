#extension GL_ARB_gpu_shader5 : enable
#version 150

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D positionTexture;
uniform sampler2D normalTexture;
uniform sampler2D noiseTexture;
uniform vec3 samples[64];
uniform float radius;
uniform float bias;

void main()
{
    vec3 fragPos = texture(positionTexture, texCoord).xyz;
    vec3 normal = normalize(texture(normalTexture, texCoord).xyz);
    vec3 randomVec = normalize(texture(noiseTexture, texCoord * 4.0).xyz);
    
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    float occlusion = 0.0;
    float intensity = 1.5;
    float baseAO = 0.3;
    
    for(int i = 0; i < 64; ++i)
    {
        vec3 samplePos = TBN * samples[i];
        samplePos = fragPos + samplePos * radius;
        
        vec4 offset = vec4(samplePos, 1.0);
        offset = offset * 0.5 + 0.5;
        
        float sampleDepth = texture(positionTexture, offset.xy).z;
        float difference = abs(fragPos.z - sampleDepth);
        
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(difference));
        float weight = step(sampleDepth, samplePos.z - bias);
        
        occlusion = occlusion + (rangeCheck * weight);
    }
    
    occlusion = 1.0 - (occlusion / 64.0);
    float ao = occlusion * intensity + baseAO;
    ao = clamp(ao, 0.0, 1.0);
    
    fragColor = vec4(ao, ao, ao, 1.0);
}