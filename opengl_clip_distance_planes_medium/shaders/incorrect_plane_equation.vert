#version 330 core

in vec3 position;

uniform mat4 modelViewProjection;
uniform vec4 clipPlane;

out float gl_ClipDistance[1];

void main()
{
    gl_Position = modelViewProjection * vec4(position, 1.0);
    
    // INCORRECT: Missing the w component of the plane equation
    // This only uses the xyz components of clipPlane and doesn't account for the homogeneous coordinate
    gl_ClipDistance[0] = dot(clipPlane.xyz, position);
}