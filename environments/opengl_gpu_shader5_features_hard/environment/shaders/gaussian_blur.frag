#extension GL_ARB_gpu_shader5 : enable
#version 150

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D inputTexture;
uniform float weights[5];
uniform vec2 direction;
uniform float blurRadius;

void main()
{
    vec3 color = vec3(0.0);
    float totalWeight = 0.0;
    
    for (int i = -2; i <= 2; i++)
    {
        float offset = float(i) * blurRadius;
        vec2 sampleCoord = texCoord + offset * direction;
        vec4 sample = texture(inputTexture, sampleCoord);
        
        int weightIndex = i + 2;
        float weight = weights[weightIndex];
        
        color = color + sample.rgb * weight;
        totalWeight = totalWeight + weight;
    }
    
    color = color / totalWeight;
    
    fragColor = vec4(color, 1.0);
}