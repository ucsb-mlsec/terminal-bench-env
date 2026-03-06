#version 450

// Post-processing fragment shader with multiple critical issues

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

// Texture samplers for post-processing
layout(set = 0, binding = 0) uniform sampler2D colorTexture;
layout(set = 0, binding = 1) uniform sampler2D depthTexture;
layout(set = 0, binding = 2) uniform sampler2D normalTexture;

// CRITICAL ISSUE: Subpass inputs are not compatible with VK_EXT_shader_object
// Shader objects don't support subpass dependencies the same way render passes do
layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput inputAttachment0;
layout(input_attachment_index = 1, set = 1, binding = 1) uniform subpassInput inputAttachment1;

// Push constants for parameters
layout(push_constant) uniform PushConstants {
    float exposure;
    float gamma;
    int effectMode;
} pushConstants;

// CRITICAL ISSUE: Attempting to use framebuffer pixel local storage
// This assumes fixed framebuffer configuration incompatible with dynamic rendering
layout(location = 0) __pixel_localEXT FragData {
    vec4 accumulationBuffer;
    vec4 revealageBuffer;
} pixelLocal;

void main() {
    // SYNTAX ERROR: undeclared variable 'baseColor'
    vec4 color = texture(colorTexture, texCoord);
    float depth = texture(depthTexture, texCoord).r;
    vec3 normal = texture(normalTexture, texCoord).rgb;
    
    // CRITICAL ISSUE: Reading from subpass input in a way that assumes
    // multi-pass rendering with specific attachment configurations
    vec4 previousPass = subpassLoad(inputAttachment0);
    vec4 blurData = subpassLoad(inputAttachment1);
    
    // SYNTAX ERROR: type mismatch - trying to add vec4 to vec3
    vec3 combined = color + previousPass;
    
    // CRITICAL ISSUE: Accessing pixel local storage that requires
    // specific framebuffer setup incompatible with shader objects
    vec4 accumulated = pixelLocal.accumulationBuffer;
    color = mix(color, accumulated, 0.5);
    
    // SYNTAX ERROR: undeclared variable 'bloomColor'
    vec3 finalColor = color.rgb + bloomColor * 0.3;
    
    // Tone mapping
    finalColor = vec3(1.0) - exp(-finalColor * pushConstants.exposure);
    finalColor = pow(finalColor, vec3(1.0 / pushConstants.gamma));
    
    outColor = vec4(finalColor, 1.0);
}