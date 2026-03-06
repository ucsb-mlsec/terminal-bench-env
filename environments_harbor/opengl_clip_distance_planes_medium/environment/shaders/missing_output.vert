#version 330 core

in vec3 position;

uniform mat4 modelViewProjection;
uniform vec4 clipPlane;

out float gl_ClipDistance[1];

void main()
{
    gl_Position = modelViewProjection * vec4(position, 1.0);
    
    // gl_ClipDistance[0] = dot(vec4(position, 1.0), clipPlane);
}