// Water Shader - Shader Model 6.6
// Renders realistic water surface with reflections and refractions

// Resource bindings for water rendering
Texture2D normalMap : register(t0);
Texture2D reflectionMap : register(s5);  // INVALID: using sampler register for texture
Texture2D refractionMap : register(t2);
TextureCube skyboxMap : register(t3);

// Sampler states
SamplerState wrapSampler : register(s0);
SamplerState clampSampler : register(s1);

// Wave simulation data
Buffer<float4> waveData[] : register(b0);  // INVALID: unbounded array with explicit register

// Constant buffers
cbuffer WaterConstants : register(b1)
{
    float4x4 viewProjection;
    float4x4 reflectionMatrix;
    float3 cameraPosition;
    float time;
    float waveScale;
    float waveSpeed;
    float fresnelBias;
    float fresnelPower;
    float4 waterColor;
    float4 shallowColor;
    float refractionStrength;
    float reflectionStrength;
    float normalStrength;
    float foamCutoff;
};

// Vertex input structure
struct VertexInput
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
};

// Pixel shader input
struct PixelInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 binormal : BINORMAL;
    float4 reflectionPos : TEXCOORD2;
};

// Vertex shader
PixelInput VSMain(VertexInput input)
{
    PixelInput output;
    
    // Apply wave displacement
    float3 worldPos = input.position;
    float wavePhase = time * waveSpeed;
    float wave1 = sin(worldPos.x * waveScale + wavePhase) * 0.1;
    float wave2 = cos(worldPos.z * waveScale * 0.7 + wavePhase * 1.3) * 0.08;
    worldPos.y += wave1 + wave2;
    
    output.worldPos = worldPos;
    output.position = mul(float4(worldPos, 1.0), viewProjection);
    output.texCoord = input.texCoord;
    
    // Transform normal
    output.normal = input.normal;
    output.tangent = input.tangent.xyz;
    output.binormal = cross(input.normal, input.tangent.xyz) * input.tangent.w;
    
    // Calculate reflection position
    output.reflectionPos = mul(float4(worldPos, 1.0), reflectionMatrix);
    
    return output;
}

// Pixel shader
float4 PSMain(PixelInput input) : SV_TARGET
{
    // Sample normal map with animation
    float2 uv1 = input.texCoord + float2(time * 0.02, time * 0.01);
    float2 uv2 = input.texCoord * 0.7 + float2(-time * 0.015, time * 0.025);
    
    float3 normal1 = normalMap.Sample(wrapSampler, uv1).xyz * 2.0 - 1.0;
    float3 normal2 = normalMap.Sample(wrapSampler, uv2).xyz * 2.0 - 1.0;
    float3 normalTS = normalize(normal1 + normal2) * normalStrength;
    
    // Transform normal to world space
    float3x3 TBN = float3x3(
        normalize(input.tangent),
        normalize(input.binormal),
        normalize(input.normal)
    );
    float3 worldNormal = normalize(mul(normalTS, TBN));
    
    // Calculate reflection and refraction UVs
    float2 reflectionUV = (input.reflectionPos.xy / input.reflectionPos.w) * 0.5 + 0.5;
    reflectionUV += worldNormal.xz * 0.05;
    
    // Sample reflection and refraction
    float3 reflection = reflectionMap.Sample(clampSampler, reflectionUV).rgb;
    float3 refraction = refractionMap.Sample(clampSampler, input.texCoord).rgb;
    
    // Fresnel effect
    float3 viewDir = normalize(cameraPosition - input.worldPos);
    float fresnel = fresnelBias + (1.0 - fresnelBias) * pow(1.0 - saturate(dot(viewDir, worldNormal)), fresnelPower);
    
    // Mix refraction with water color
    float depth = saturate(input.worldPos.y * 0.1);
    float3 waterTint = lerp(shallowColor.rgb, waterColor.rgb, depth);
    refraction = lerp(refraction, waterTint, 0.3);
    
    // Combine reflection and refraction
    float3 finalColor = lerp(refraction * refractionStrength, reflection * reflectionStrength, fresnel);
    
    // Add specular highlights
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float3 halfVec = normalize(viewDir + lightDir);
    float specular = pow(saturate(dot(worldNormal, halfVec)), 128.0);
    finalColor += specular * 0.5;
    
    // Skybox reflection for distant water
    float3 reflectDir = reflect(-viewDir, worldNormal);
    float3 skyReflection = skyboxMap.Sample(wrapSampler, reflectDir).rgb;
    finalColor = lerp(finalColor, skyReflection, fresnel * 0.3);
    
    return float4(finalColor, 1.0);
}