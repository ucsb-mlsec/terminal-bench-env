#version 330 core

in vec3 position;
in vec3 normal;

out float gl_ClipDistance[1];
out vec3 fragNormal;

uniform mat4 modelViewProjection;
uniform mat4 normalMatrix;
uniform vec4 clipPlane;

void main()
{
    gl_Position = modelViewProjection * vec4(position, 1.0);
    fragNormal = mat3(normalMatrix) * normal;
    gl_ClipDistance[0] = dot(vec4(position, 1.0), clipPlane);
}