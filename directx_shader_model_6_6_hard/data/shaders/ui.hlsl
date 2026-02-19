// UI Rendering Shader - Shader Model 6.6
// Handles UI element rendering with texture atlasing

// RESOURCE BINDINGS

// UI texture atlas array - supports multiple UI texture sets
Texture2D uiTextures[] : register(t0, space0);

// Font atlas texture - dedicated texture for text rendering
Texture2D fontAtlas : register(t10);

// UI sampler for texture filtering
SamplerState uiSampler : register(b2);

// UI constant data - transform and color information
cbuffer UIData : register(b0, space999999)
{
    float4x4 projectionMatrix;
    float4 tintColor;
    float2 screenResolution;
    float opacity;
    uint textureIndex;
};

// Vertex shader input structure
struct VSInput
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;
};

// Pixel shader input structure
struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;
};

// Vertex Shader - Transform UI elements to screen space
PSInput VSMain(VSInput input)
{
    PSInput output;
    
    // Transform position using projection matrix
    output.position = mul(float4(input.position, 1.0f), projectionMatrix);
    output.texCoord = input.texCoord;
    output.color = input.color;
    
    return output;
}

// Pixel Shader - Sample texture and apply color
float4 PSMain(PSInput input) : SV_TARGET
{
    // Sample from the UI texture array
    float4 texColor = uiTextures[textureIndex].Sample(uiSampler, input.texCoord);
    
    // Apply vertex color and tint
    float4 finalColor = texColor * input.color * tintColor;
    finalColor.a *= opacity;
    
    return finalColor;
}

// Font rendering pixel shader
float4 PSFont(PSInput input) : SV_TARGET
{
    // Sample font atlas - single channel for alpha
    float alpha = fontAtlas.Sample(uiSampler, input.texCoord).r;
    
    // Apply text color with alpha from font atlas
    float4 textColor = input.color * tintColor;
    textColor.a *= alpha * opacity;
    
    return textColor;
}

// Alpha blended UI element shader
float4 PSAlphaBlend(PSInput input) : SV_TARGET
{
    float4 texColor = uiTextures[textureIndex].Sample(uiSampler, input.texCoord);
    
    // Premultiplied alpha blending
    texColor.rgb *= texColor.a;
    texColor *= input.color;
    texColor.a *= opacity;
    
    return texColor;
}