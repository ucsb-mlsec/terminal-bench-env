#include <metal_stdlib>
#include <metal_common>

using namespace metal;

kernel void imageProcessingKernel(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    device float* brightnessParams [[buffer(0)]],
    constant float& contrastFactor [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) {
        return;
    }
    
    float4 inputColor = inputTexture.read(gid);
    
    float brightness = brightnessParams[0];
    float contrast = contrastFactor;
    
    float4 adjustedColor;
    adjustedColor.rgb = (inputColor.rgb - 0.5) * contrast + 0.5 + brightness;
    adjustedColor.rgb = clamp(adjustedColor.rgb, 0.0, 1.0);
    adjustedColor.a = inputColor.a;
    
    outputTexture.write(adjustedColor, gid);
}