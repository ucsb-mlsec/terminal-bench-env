#include "lighting/phong.glsl"
#include  "common/defines.glsl"

precision highp float;

in vec3 fragPosition;
in vec3 fragNormal;
in vec2 fragTexCoord;

out vec4 FragColor;

uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform sampler2D diffuseTexture;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(viewPosition - fragPosition);
    
    vec3 phongColor = calculatePhong(fragPosition, normal, viewDir, lightPosition);
    vec4 texColor = texture(diffuseTexture, fragTexCoord);
    
    FragColor = vec4(phongColor * texColor.rgb, 1.0);
}