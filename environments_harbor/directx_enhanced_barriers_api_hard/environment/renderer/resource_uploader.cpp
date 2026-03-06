#include <d3d12.h>
#include "d3dx12.h"
#include <vector>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

class ResourceUploader
{
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12GraphicsCommandList7> m_commandList7;
    ComPtr<ID3D12Resource> m_uploadBuffer;
    SIZE_T m_uploadBufferSize;

public:
    ResourceUploader(ID3D12Device* device, ID3D12GraphicsCommandList* commandList)
        : m_device(device)
        , m_commandList(commandList)
        , m_uploadBufferSize(0)
    {
        // Query for enhanced barrier support
        m_commandList->QueryInterface(IID_PPV_ARGS(&m_commandList7));
    }

    ~ResourceUploader()
    {
        // Cleanup resources
    }

    // Upload buffer data from CPU to GPU using legacy barriers
    HRESULT UploadBufferData(ID3D12Resource* destBuffer, const void* data, SIZE_T dataSize)
    {
        // Create upload heap
        D3D12_HEAP_PROPERTIES uploadHeapProps = {};
        uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        uploadHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        uploadHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = dataSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        ComPtr<ID3D12Resource> uploadBuffer;
        HRESULT hr = m_device->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&uploadBuffer));

        if (FAILED(hr))
            return hr;

        // Map and copy data
        void* mappedData = nullptr;
        hr = uploadBuffer->Map(0, nullptr, &mappedData);
        if (FAILED(hr))
            return hr;

        memcpy(mappedData, data, dataSize);
        uploadBuffer->Unmap(0, nullptr);

        // Transition destination buffer to copy dest state
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            destBuffer,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST);
        m_commandList->ResourceBarrier(1, &barrier);

        // Copy data from upload buffer to destination
        m_commandList->CopyBufferRegion(destBuffer, 0, uploadBuffer.Get(), 0, dataSize);

        // Transition back to common state
        barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            destBuffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_GENERIC_READ);
        m_commandList->ResourceBarrier(1, &barrier);

        return S_OK;
    }

    // Upload texture data using legacy barriers
    HRESULT UploadTextureData(ID3D12Resource* destTexture, const void* data, 
                              UINT width, UINT height, DXGI_FORMAT format)
    {
        // Calculate data size
        UINT bytesPerPixel = 4; // Assuming RGBA8
        UINT rowPitch = width * bytesPerPixel;
        UINT textureSize = rowPitch * height;

        // Create upload buffer
        D3D12_HEAP_PROPERTIES uploadHeapProps = {};
        uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC uploadBufferDesc = {};
        uploadBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        uploadBufferDesc.Width = textureSize;
        uploadBufferDesc.Height = 1;
        uploadBufferDesc.DepthOrArraySize = 1;
        uploadBufferDesc.MipLevels = 1;
        uploadBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        uploadBufferDesc.SampleDesc.Count = 1;
        uploadBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        ComPtr<ID3D12Resource> uploadBuffer;
        HRESULT hr = m_device->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &uploadBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&uploadBuffer));

        if (FAILED(hr))
            return hr;

        // Map and copy texture data
        void* mappedData = nullptr;
        hr = uploadBuffer->Map(0, nullptr, &mappedData);
        if (FAILED(hr))
            return hr;

        memcpy(mappedData, data, textureSize);
        uploadBuffer->Unmap(0, nullptr);

        // Transition texture to copy destination state
        D3D12_RESOURCE_BARRIER copyBarrier = {};
        copyBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        copyBarrier.Transition.pResource = destTexture;
        copyBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        copyBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        copyBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_commandList->ResourceBarrier(1, &copyBarrier);

        // Setup texture copy location
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
        footprint.Offset = 0;
        footprint.Footprint.Format = format;
        footprint.Footprint.Width = width;
        footprint.Footprint.Height = height;
        footprint.Footprint.Depth = 1;
        footprint.Footprint.RowPitch = rowPitch;

        D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
        srcLocation.pResource = uploadBuffer.Get();
        srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLocation.PlacedFootprint = footprint;

        D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
        dstLocation.pResource = destTexture;
        dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLocation.SubresourceIndex = 0;

        m_commandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);

        return S_OK;
    }

    // Enhanced barrier version for modern path
    HRESULT UploadTextureDataEnhanced(ID3D12Resource* destTexture, const void* data,
                                      UINT width, UINT height)
    {
        if (!m_commandList7)
            return E_NOINTERFACE;

        // Setup enhanced texture barrier
        D3D12_TEXTURE_BARRIER textureBarrier = {};
        textureBarrier.SyncBefore = D3D12_BARRIER_SYNC_COPY;
        textureBarrier.SyncAfter = D3D12_BARRIER_SYNC_ALL;
        textureBarrier.AccessBefore = D3D12_BARRIER_ACCESS_COPY_DEST;
        textureBarrier.AccessAfter = D3D12_BARRIER_ACCESS_COMMON;
        textureBarrier.LayoutBefore = D3D12_BARRIER_LAYOUT_COPY_DEST;
        textureBarrier.LayoutAfter = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON;
        textureBarrier.pResource = destTexture;
        textureBarrier.Subresources.IndexOrFirstMipLevel = 0;
        textureBarrier.Subresources.NumMipLevels = 1;

        D3D12_BARRIER_GROUP barrierGroup = {};
        barrierGroup.Type = D3D12_BARRIER_TYPE_TEXTURE;
        barrierGroup.NumBarriers = 1;
        barrierGroup.pTextureBarriers = &textureBarrier;

        m_commandList7->Barrier(1, &barrierGroup);

        return S_OK;
    }
};