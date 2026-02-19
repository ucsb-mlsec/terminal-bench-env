#version 400 core

in vec3 position;

uniform mat4 modelMatrix;
uniform mat4 viewProjectionMatrix;
uniform vec4 clipPlanes[3];

out float gl_ClipDistance[3];

void main()
{
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    gl_Position = viewProjectionMatrix * worldPos;
    
    gl_ClipDistance[0] = dot(worldPos, clipPlanes[0]);
    gl_ClipDistance[1] = dot(worldPos, clipPlanes[1]);
    gl_ClipDistance[2] = dot(worldPos, clipPlanes[2]);
}