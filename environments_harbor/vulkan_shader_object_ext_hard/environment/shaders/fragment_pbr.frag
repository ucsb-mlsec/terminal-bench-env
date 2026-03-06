#version 450
#extension GL_EXT_tessellation_shader : require
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D albedoMap;
layout(set = 0, binding = 1) uniform sampler2D normalMap;
layout(set = 0, binding = 1) uniform sampler2D roughnessMap;
layout(set = 2, binding = 0) uniform LightData {
    vec3 lightPos;
    vec3 lightColor;
    float intensity;
} light;

layout(push_constant) uniform PushConstants {
    mat4 view;
    vec3 cameraPos;
} pc;

void main() {
    vec3 albedo = texture(albedoMap, fragTexCoord).rgb;
    vec3 normal = normalize(fragNormal);
    
    vec3 lightDir = normalize(light.lightPos - fragWorldPos);
    vec3 viewDir = normalize(cameraPosition - fragWorldPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    
    float roughness = texture(roughnessMap, fragTexCoord).r;
    float metallic = metallicValue;
    
    float NdotL = max(dot(normal, lightDir), 0.0);
    float NdotH = max(dot(normal, halfDir), 0.0);
    
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 specular = F0 * pow(NdotH, roughness * 256.0);
    
    vec3 ambient = vec3(0.03) * albedo;
    vec3 diffuse = albedo * NdotL * light.lightColor;
    
    outColor = vec4(ambient + diffuse + specular, 1.0);
    
    gl_FragDepth = gl_FragCoord.z * 0.5 + 0.5;
}