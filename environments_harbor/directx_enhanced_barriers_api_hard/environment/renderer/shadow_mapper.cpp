#include <d3d12.h>
#include "d3dx12.h"
#include <vector>
#include <cstdint>

// Shadow mapping system for cascaded shadow maps
// Uses Enhanced Barriers API for resource synchronization

constexpr uint32_t MAX_SHADOW_CASCADES = 4;
constexpr uint32_t SHADOW_MAP_RESOLUTION = 2048;

struct ShadowCascade {
    ID3D12Resource* depthBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
    float splitDistance;
};

class ShadowMapper {
private:
    std::vector<ShadowCascade> m_cascades;
    ID3D12Device10* m_device;
    ID3D12DescriptorHeap* m_dsvHeap;
    ID3D12DescriptorHeap* m_srvHeap;

public:
    ShadowMapper(ID3D12Device10* device) : m_device(device) {
        m_cascades.resize(MAX_SHADOW_CASCADES);
        InitializeShadowCascades();
    }

    void InitializeShadowCascades() {
        // Initialize depth buffers for each cascade
        D3D12_RESOURCE_DESC depthDesc = {};
        depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        depthDesc.Width = SHADOW_MAP_RESOLUTION;
        depthDesc.Height = SHADOW_MAP_RESOLUTION;
        depthDesc.DepthOrArraySize = 1;
        depthDesc.MipLevels = 1;
        depthDesc.Format = DXGI_FORMAT_D32_FLOAT;
        depthDesc.SampleDesc.Count = 1;
        depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        for (uint32_t i = 0; i < MAX_SHADOW_CASCADES; ++i) {
            D3D12_HEAP_PROPERTIES heapProps = {};
            heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

            m_device->CreateCommittedResource(
                &heapProps,
                D3D12_HEAP_FLAG_NONE,
                &depthDesc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                nullptr,
                IID_PPV_ARGS(&m_cascades[i].depthBuffer)
            );
        }
    }

    void RenderShadowCascades(ID3D12GraphicsCommandList7* commandList,
                             const std::vector<ID3D12Resource*>& sceneGeometry) {
        // Transition all shadow map depth buffers to writable state
        D3D12_TEXTURE_BARRIER textureBarriers[MAX_SHADOW_CASCADES] = {};
        
        for (uint32_t cascade = 0; cascade < MAX_SHADOW_CASCADES; ++cascade) {
            textureBarriers[cascade].SyncBefore = D3D12_BARRIER_SYNC_PIXEL_SHADING;
            textureBarriers[cascade].SyncAfter = D3D12_BARRIER_SYNC_DEPTH_STENCIL;
            textureBarriers[cascade].AccessBefore = D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
            textureBarriers[cascade].AccessAfter = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
            textureBarriers[cascade].LayoutBefore = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
            textureBarriers[cascade].LayoutAfter = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
            textureBarriers[cascade].pResource = m_cascades[cascade].depthBuffer;
            textureBarriers[cascade].Subresources.IndexOrFirstMipLevel = 0;
            textureBarriers[cascade].Subresources.NumMipLevels = 1;
            textureBarriers[cascade].Flags = D3D12_TEXTURE_BARRIER_FLAG_NONE;
        }

        D3D12_BARRIER_GROUP barrierGroup = {};
        barrierGroup.Type = D3D12_BARRIER_TYPE_TEXTURE;
        barrierGroup.NumBarriers = MAX_SHADOW_CASCADES;
        barrierGroup.pTextureBarriers = textureBarriers;

        commandList->Barrier(1, &barrierGroup);

        // Render shadow maps for each cascade
        for (uint32_t cascade = 0; cascade < MAX_SHADOW_CASCADES; ++cascade) {
            // Clear depth buffer
            commandList->ClearDepthStencilView(
                m_cascades[cascade].dsvHandle,
                D3D12_CLEAR_FLAG_DEPTH,
                1.0f,
                0,
                0,
                nullptr
            );

            // Set viewport and scissor for this cascade
            D3D12_VIEWPORT viewport = {};
            viewport.Width = static_cast<float>(SHADOW_MAP_RESOLUTION);
            viewport.Height = static_cast<float>(SHADOW_MAP_RESOLUTION);
            viewport.MinDepth = 0.0f;
            viewport.MaxDepth = 1.0f;
            commandList->RSSetViewports(1, &viewport);

            D3D12_RECT scissorRect = { 0, 0, SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION };
            commandList->RSSetScissorRects(1, &scissorRect);

            // Set render target
            commandList->OMSetRenderTargets(0, nullptr, FALSE, &m_cascades[cascade].dsvHandle);

            // Draw scene geometry from light perspective
            for (auto* geometry : sceneGeometry) {
                // Geometry drawing calls would go here
            }
        }
    }

    void TransitionShadowMapsForSampling(ID3D12GraphicsCommandList7* commandList) {
        // Transition shadow maps from depth write to shader resource for sampling
        D3D12_TEXTURE_BARRIER readBarriers[MAX_SHADOW_CASCADES] = {};

        for (uint32_t i = 0; i < MAX_SHADOW_CASCADES; ++i) {
            readBarriers[i].SyncBefore = D3D12_BARRIER_SYNC_DEPTH_STENCIL;
            readBarriers[i].SyncAfter = D3D12_BARRIER_SYNC_PIXEL_SHADING;
            readBarriers[i].AccessBefore = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
            readBarriers[i].AccessAfter = D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
            readBarriers[i].LayoutBefore = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
            readBarriers[i].LayoutAfter = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
            readBarriers[i].pResource = m_cascades[i].depthBuffer;
            readBarriers[i].Subresources.IndexOrFirstMipLevel = 0;
            readBarriers[i].Subresources.NumMipLevels = 1;
            readBarriers[i].Flags = D3D12_TEXTURE_BARRIER_FLAG_NONE;
        }

        D3D12_BARRIER_GROUP readBarrierGroup = {};
        readBarrierGroup.Type = D3D12_BARRIER_TYPE_TEXTURE;
        readBarrierGroup.NumBarriers = MAX_SHADOW_CASCADES;
        readBarrierGroup.pTextureBarriers = readBarriers;

        commandList->Barrier(1, &readBarrierGroup);
    }

    ~ShadowMapper() {
        for (auto& cascade : m_cascades) {
            if (cascade.depthBuffer) {
                cascade.depthBuffer->Release();
            }
        }
    }
};