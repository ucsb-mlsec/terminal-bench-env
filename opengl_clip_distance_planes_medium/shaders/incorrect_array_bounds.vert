#version 330 core

in vec3 position;

uniform mat4 modelViewProjection;
uniform vec4 clipPlanes[4];

out float gl_ClipDistance[2];

void main()
{
    gl_Position = modelViewProjection * vec4(position, 1.0);
    
    gl_ClipDistance[0] = dot(vec4(position, 1.0), clipPlanes[0]);
    gl_ClipDistance[1] = dot(vec4(position, 1.0), clipPlanes[1]);
    gl_ClipDistance[3] = dot(vec4(position, 1.0), clipPlanes[3]);
}