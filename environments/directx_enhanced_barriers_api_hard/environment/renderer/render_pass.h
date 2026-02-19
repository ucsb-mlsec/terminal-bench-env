#ifndef RENDERER_RENDER_PASS_H
#define RENDERER_RENDER_PASS_H

#include <d3d12.h>
#include "d3dx12.h"

namespace Renderer {

// Forward declarations
class RenderTarget;
class DepthBuffer;

// Inline helper function for transitioning render target to render target state
inline void TransitionToRenderTarget(ID3D12GraphicsCommandList* commandList, ID3D12Resource* resource, D3D12_RESOURCE_STATES beforeState)
{
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        resource,
        beforeState,
        D3D12_RESOURCE_STATE_RENDER_TARGET
    );
    commandList->ResourceBarrier(1, &barrier);
}

// Inline helper function for transitioning resource to shader resource state
inline void TransitionToShaderResource(ID3D12GraphicsCommandList* commandList, ID3D12Resource* resource, D3D12_RESOURCE_STATES beforeState)
{
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        resource,
        beforeState,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );
    commandList->ResourceBarrier(1, &barrier);
}

// Enhanced barrier helper using new API
inline void TransitionToRenderTargetEnhanced(ID3D12GraphicsCommandList7* commandList, ID3D12Resource* resource)
{
    D3D12_TEXTURE_BARRIER textureBarrier = {};
    textureBarrier.SyncBefore = D3D12_BARRIER_SYNC_RENDER_TARGET;
    textureBarrier.SyncAfter = D3D12_BARRIER_SYNC_RENDER_TARGET;
    textureBarrier.AccessBefore = D3D12_BARRIER_ACCESS_NO_ACCESS;
    textureBarrier.AccessAfter = D3D12_BARRIER_ACCESS_RENDER_TARGET;
    textureBarrier.LayoutBefore = D3D12_BARRIER_LAYOUT_COMMON;
    textureBarrier.LayoutAfter = D3D12_BARRIER_LAYOUT_RENDER_TARGET;
    textureBarrier.pResource = resource;
    textureBarrier.Subresources.IndexOrFirstMipLevel = 0;
    textureBarrier.Subresources.NumMipLevels = 1;
    
    D3D12_BARRIER_GROUP barrierGroup = {};
    barrierGroup.Type = D3D12_BARRIER_TYPE_TEXTURE;
    barrierGroup.NumBarriers = 1;
    barrierGroup.pTextureBarriers = &textureBarrier;
    
    commandList->Barrier(1, &barrierGroup);
}

// RenderPass class for managing rendering operations
class RenderPass
{
public:
    RenderPass();
    ~RenderPass();
    
    // Initialize render pass with targets
    bool Initialize(ID3D12Device* device, UINT width, UINT height);
    
    // Begin render pass - transitions resources to appropriate states
    void Begin(ID3D12GraphicsCommandList* commandList);
    
    // End render pass - transitions resources back
    void End(ID3D12GraphicsCommandList* commandList);
    
    // Bind render targets
    void BindRenderTargets(ID3D12GraphicsCommandList* commandList);
    
    // Clear render targets and depth buffer
    void Clear(ID3D12GraphicsCommandList* commandList);
    
    // Get render target resource
    ID3D12Resource* GetRenderTarget() const { return m_renderTarget; }
    
    // Get depth buffer resource
    ID3D12Resource* GetDepthBuffer() const { return m_depthBuffer; }
    
private:
    ID3D12Resource* m_renderTarget;
    ID3D12Resource* m_depthBuffer;
    D3D12_RESOURCE_STATES m_renderTargetState;
    D3D12_RESOURCE_STATES m_depthBufferState;
    UINT m_width;
    UINT m_height;
    
    // Internal helper for state management
    void TransitionResources(ID3D12GraphicsCommandList* commandList, bool toRenderState);
};

} // namespace Renderer

#endif // RENDERER_RENDER_PASS_H