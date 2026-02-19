#version 330 core

// Input vertex attributes
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

// Output to fragment shader
out vec3 ourColor;

// Transformation matrix (optional)
uniform mat4 mvp;

void main()
{
    // Transform vertex position
    gl_Position = vec4(aPos, 1.0);
    
    // Pass color to fragment shader
    ourColor = aColor;
}