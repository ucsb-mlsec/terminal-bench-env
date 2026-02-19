#version 330 core

in vec3 position;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec4 clipPlane;

out float gl_ClipDistance[1];

void main()
{
    vec4 viewPos = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * viewPos;
    gl_ClipDistance[0] = dot(viewPos, clipPlane);
}