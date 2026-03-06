#version 330 core

in vec3 position;
in float gl_ClipDistance[1];

uniform mat4 modelViewProjection;
uniform vec4 clipPlane;

void main()
{
    gl_Position = modelViewProjection * vec4(position, 1.0);
    gl_ClipDistance[0] = dot(vec4(position, 1.0), clipPlane);
}