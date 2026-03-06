#version 400 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoords;
layout (location = 2) in vec3 normal;

out vec2 texCoord;
out vec3 fragNormal;
out vec3 fragPosition;
out float height;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Sample height from heightmap
    float heightValue = texture(heightMap, texCoords).r;
    
    // Calculate displaced position
    vec3 displacedPos = position;
    displacedPos.y += heightValue * 10.0;
    
    // Transform position
    vec4 worldPos = model * vec4(displacedPos, 1.0);
    fragPosition = worldPos.xyz;
    gl_Position = projection * view * worldPos;
    
    // Pass texture coordinates (wrong type assignment)
    texCoord = vec3(texCoords, heightValue);
    
    // Transform normal
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    fragNormal = normalize(normalMatrix * normal);
    
    // Store height for fragment shader
    height = heightValue;
    
    // Use GLSL 4.0+ function
    uint packedColor = packUnorm4x8(vec4(heightValue, heightValue, heightValue, 1.0));
}