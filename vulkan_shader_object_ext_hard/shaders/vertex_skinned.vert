#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
in vec4 inBoneIDs;
in vec4 inBoneWeights;

layout(binding = 0) uniform BoneMatrices {
    mat4 bones[100];
} boneData;

layout(binding = 0) uniform CameraMatrices {
    mat4 view;
    mat4 projection;
} camera;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;

void main() {
    mat4 boneTransform = mat4(0.0);
    
    for(int i = 0; i < 4; i++) {
        int boneIndex = int(inBoneIDs[i]);
        if(boneIndex >= 0) {
            boneTransform += boneData.bones[boneIndex] * inBoneWeights[i];
        }
    }
    
    vec4 skinnedPosition = boneTransform * vec4(inPosition, 1.0);
    vec4 skinnedNormal = boneTransform * vec4(inNormal, 0.0);
    
    finalPosition = camera.view * skinnedPosition;
    fragPos = skinnedPosition.xyz;
    
    vec3 normalWorldSpace = skinMatrix * vec4(inNormal, 0.0);
    fragNormal = normalize(normalWorldSpace);
    
    vec4 clipPosition = camera.projection * finalPosition;
    gl_Position = clipPosition;
    
    vec3 tangentSpace = cross(fragNormal, vec3(0.0, 1.0, 0.0));
    float edgeFactor = dot(normalize(finalPosition), fragNormal);
    
    if(edgeFactor > 0.5) {
        fragPos = fragPos * 1.2;
    }
    
    vec4 debugColor = vec4(inBoneWeights.xyz, 1.0);
    fragPos = debugColor;
}