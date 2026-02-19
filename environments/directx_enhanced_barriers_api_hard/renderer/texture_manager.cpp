#include <d3d12.h>
#include "d3dx12.h"
#include <vector>
#include <string>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace Renderer {

class TextureManager {
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12GraphicsCommandList7> m_commandList7;
    std::vector<ComPtr<ID3D12Resource>> m_textures;

public:
    TextureManager(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) 
        : m_device(device), m_commandList(cmdList) {
        // Query for enhanced barriers support
        cmdList->QueryInterface(IID_PPV_ARGS(&m_commandList7));
    }

    // Load a texture from file and transition to shader resource state
    HRESULT LoadTexture(const std::wstring& filename, UINT textureIndex) {
        // Create texture resource (simplified for this example)
        ComPtr<ID3D12Resource> texture;
        D3D12_RESOURCE_DESC textureDesc = {};
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.Width = 1024;
        textureDesc.Height = 1024;
        textureDesc.DepthOrArraySize = 1;
        textureDesc.MipLevels = 1;
        textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
        HRESULT hr = m_device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &textureDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&texture));

        if (FAILED(hr)) {
            return hr;
        }

        // Upload texture data (simplified - normally would use upload heap)
        // After upload, transition from COPY_DEST to PIXEL_SHADER_RESOURCE

        // Legacy barrier pattern - transition after copy
        CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            texture.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        
        m_commandList->ResourceBarrier(1, &barrier);

        m_textures.push_back(texture);
        return S_OK;
    }

    // Update a texture region with new data
    void UpdateTextureRegion(UINT textureIndex, const void* data, size_t dataSize) {
        if (textureIndex >= m_textures.size()) {
            return;
        }

        ID3D12Resource* texture = m_textures[textureIndex].Get();

        // Transition texture to copy destination for update
        CD3DX12_RESOURCE_BARRIER toWrite = CD3DX12_RESOURCE_BARRIER::Transition(
            texture,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST);

        m_commandList->ResourceBarrier(1, &toWrite);

        // Perform update operation here (simplified)
        // In real code, would copy data from upload buffer

        // Transition back to shader resource
        D3D12_RESOURCE_BARRIER transitionBarrier = {};
        transitionBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transitionBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        transitionBarrier.Transition.pResource = texture;
        transitionBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        transitionBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        transitionBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        m_commandList->ResourceBarrier(1, &transitionBarrier);
    }

    // Prepare texture for use as render target
    void PrepareTextureAsRenderTarget(UINT textureIndex) {
        if (textureIndex >= m_textures.size()) {
            return;
        }

        ID3D12Resource* texture = m_textures[textureIndex].Get();

        // Use enhanced barrier API for this operation
        if (m_commandList7) {
            D3D12_TEXTURE_BARRIER textureBarrier = {};
            textureBarrier.SyncBefore = D3D12_BARRIER_SYNC_PIXEL_SHADING;
            textureBarrier.SyncAfter = D3D12_BARRIER_SYNC_RENDER_TARGET;
            textureBarrier.AccessBefore = D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
            textureBarrier.AccessAfter = D3D12_BARRIER_ACCESS_RENDER_TARGET;
            textureBarrier.LayoutBefore = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
            textureBarrier.LayoutAfter = D3D12_BARRIER_LAYOUT_RENDER_TARGET;
            textureBarrier.pResource = texture;
            textureBarrier.Subresources.IndexOrFirstMipLevel = 0;
            textureBarrier.Subresources.NumMipLevels = 1;
            textureBarrier.Flags = D3D12_TEXTURE_BARRIER_FLAG_NONE;

            D3D12_BARRIER_GROUP barrierGroup = {};
            barrierGroup.Type = D3D12_BARRIER_TYPE_TEXTURE;
            barrierGroup.NumBarriers = 1;
            barrierGroup.pTextureBarriers = &textureBarrier;

            m_commandList7->Barrier(1, &barrierGroup);
        } else {
            // Fallback to legacy barriers
            D3D12_RESOURCE_BARRIER barrier = {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = texture;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            m_commandList->ResourceBarrier(1, &barrier);
        }
    }

    // Copy texture for readback operations
    void CopyTextureForReadback(UINT srcTextureIndex, ID3D12Resource* dstBuffer) {
        if (srcTextureIndex >= m_textures.size()) {
            return;
        }

        ID3D12Resource* srcTexture = m_textures[srcTextureIndex].Get();

        // Transition source texture to copy source using enhanced barriers
        if (m_commandList7) {
            D3D12_TEXTURE_BARRIER srcBarrier = {};
            srcBarrier.SyncBefore = D3D12_BARRIER_SYNC_PIXEL_SHADING;
            srcBarrier.SyncAfter = D3D12_BARRIER_SYNC_COPY;
            srcBarrier.AccessBefore = D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
            srcBarrier.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
            srcBarrier.LayoutBefore = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
            srcBarrier.LayoutAfter = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
            srcBarrier.pResource = srcTexture;
            srcBarrier.Subresources.IndexOrFirstMipLevel = 0;
            srcBarrier.Subresources.NumMipLevels = 1;

            D3D12_BARRIER_GROUP barriers = {};
            barriers.Type = D3D12_BARRIER_TYPE_TEXTURE;
            barriers.NumBarriers = 1;
            barriers.pTextureBarriers = &srcBarrier;

            m_commandList7->Barrier(1, &barriers);
        }

        // Perform copy operation
        D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
        srcLocation.pResource = srcTexture;
        srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLocation.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
        dstLocation.pResource = dstBuffer;
        dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;

        m_commandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);

        // Transition back to shader resource using legacy barrier
        D3D12_RESOURCE_BARRIER backBarrier = {};
        backBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        backBarrier.Transition.pResource = srcTexture;
        backBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        backBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        backBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        m_commandList->ResourceBarrier(1, &backBarrier);
    }

    // Batch transition multiple textures for rendering
    void BatchTransitionTextures(const std::vector<UINT>& textureIndices, D3D12_RESOURCE_STATES targetState) {
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        barriers.reserve(textureIndices.size());

        for (UINT index : textureIndices) {
            if (index < m_textures.size()) {
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition.pResource = m_textures[index].Get();
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                barrier.Transition.StateAfter = targetState;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barriers.push_back(barrier);
            }
        }

        if (!barriers.empty()) {
            // Submit all transitions in one batch
            m_commandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
        }
    }
};

} // namespace Renderer