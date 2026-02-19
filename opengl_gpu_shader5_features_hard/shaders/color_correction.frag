#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D colorTexture;
uniform float brightness;
uniform float contrast;
uniform vec3 colorTint;
uniform float saturation;

void main()
{
    vec4 texColor = texture(colorTexture, texCoord);
    vec3 color = texColor.rgb;
    
    // Apply contrast adjustment
    vec3 adjusted = color * contrast;
    adjusted = adjusted + brightness;
    
    // Apply color tinting
    vec3 tinted = adjusted * colorTint;
    vec3 offset = vec3(0.05, 0.05, 0.05);
    tinted = tinted + offset;
    
    // Desaturation calculation
    float gray = dot(tinted, vec3(0.299, 0.587, 0.114));
    vec3 grayColor = vec3(gray);
    vec3 finalColor = mix(grayColor, tinted, saturation);
    
    fragColor = vec4(finalColor, texColor.a);
}