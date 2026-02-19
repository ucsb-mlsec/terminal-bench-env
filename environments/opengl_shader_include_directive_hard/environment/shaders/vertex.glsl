#include "common/defines.glsl"
#include   "common/utils.glsl"
#include "lighting/phong.glsl"

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vFragPos;

void main() {
    vFragPos = vec3(MODEL_MATRIX * vec4(aPosition, 1.0));
    vNormal = mat3(transpose(inverse(MODEL_MATRIX))) * aNormal;
    vTexCoord = aTexCoord;
    
    gl_Position = PROJECTION_MATRIX * VIEW_MATRIX * vec4(vFragPos, 1.0);
}