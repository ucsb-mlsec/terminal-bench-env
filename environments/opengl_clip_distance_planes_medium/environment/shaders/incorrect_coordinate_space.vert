#version 330 core

in vec3 position;

uniform mat4 modelViewProjection;
uniform vec4 clipPlane;

out float gl_ClipDistance[1];

void main()
{
    vec4 clipSpacePos = modelViewProjection * vec4(position, 1.0);
    gl_Position = clipSpacePos;
    
    // ERROR: Using clip space position with world space plane equation
    // This is incorrect because clipPlane is in world space but we're dotting it
    // with a clip space position. The coordinate spaces don't match.
    gl_ClipDistance[0] = dot(clipSpacePos, clipPlane);
}