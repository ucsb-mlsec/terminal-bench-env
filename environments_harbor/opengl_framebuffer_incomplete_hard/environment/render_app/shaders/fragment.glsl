#version 330 core

in vec3 ourColor;
in vec2 TexCoord;

out vec4 FragColor;

void main()
{
    // Output the interpolated color with full opacity
    FragColor = vec4(ourColor, 1.0);
    
    // Apply a slight gamma correction for better visual output
    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/2.2));
}