#version 400
#extension GL_ARB_gpu_shader5 : enable

// Fragment shader for complex texture gather operations
// Multi-pass rendering support required

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D tex0;
uniform sampler2D tex1;

// Unused precise variable - should be removed or used
precise float unusedPrecise = 3.14159;

void main()
{
    // ISSUE: Missing precision qualifier on gather result
    vec4 gathered0 = textureGather(tex0, vTexCoord);
    
    // ISSUE: Missing precision qualifier on gather offset result
    vec4 gathered1 = textureGatherOffset(tex1, vTexCoord, ivec2(1,1));
    
    // Combine gathered values
    vec4 combined = gathered0 + gathered1;
    
    // Apply some filtering
    float avg0 = (gathered0.x + gathered0.y + gathered0.z + gathered0.w) * 0.25;
    float avg1 = (gathered1.x + gathered1.y + gathered1.z + gathered1.w) * 0.25;
    
    // Mix results
    vec4 result = mix(gathered0, gathered1, avg0);
    
    // Final color output
    fragColor = result * vec4(avg1);
}