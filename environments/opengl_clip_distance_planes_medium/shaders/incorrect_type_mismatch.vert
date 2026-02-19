#version 330 core

out float gl_ClipDistance[1];

in vec3 position;

uniform mat4 modelViewProjection;
uniform vec4 clipPlane;

void main()
{
    gl_Position = modelViewProjection * vec4(position, 1.0);
    gl_ClipDistance[0] = dot(position, clipPlane);
}