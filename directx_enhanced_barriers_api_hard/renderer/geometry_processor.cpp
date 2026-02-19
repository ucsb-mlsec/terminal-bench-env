#include <d3d12.h>
#include "d3dx12.h"
#include <wrl/client.h>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

// Geometry Processor for DirectX 12 Rendering Pipeline
// This module handles vertex buffer and index buffer operations
// including resource state transitions and synchronization

class GeometryProcessor {
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12GraphicsCommandList7> m_commandList7;
    ComPtr<ID3D12Resource> m_vertexBuffer;
    ComPtr<ID3D12Resource> m_indexBuffer;
    ComPtr<ID3D12Resource> m_uploadBuffer;
    
public:
    GeometryProcessor(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
        m_device = device;
        m_commandList = cmdList;
        
        // Try to get enhanced barrier interface
        cmdList->QueryInterface(IID_PPV_ARGS(&m_commandList7));
    }
    
    // Upload vertex data to GPU buffer
    void UploadVertexData(const void* vertexData, size_t dataSize) {
        // First, transition the vertex buffer to copy destination state
        // Legacy barrier API: ResourceBarrier call
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_vertexBuffer.Get(),
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_RESOURCE_STATE_COPY_DEST
        );
        
        m_commandList->ResourceBarrier(1, &barrier);
        
        // Copy data from upload buffer to vertex buffer
        m_commandList->CopyBufferRegion(
            m_vertexBuffer.Get(), 0,
            m_uploadBuffer.Get(), 0,
            dataSize
        );
        
        // Transition back to vertex buffer state
        // Legacy barrier API: Another ResourceBarrier call
        auto barrierBack = CD3DX12_RESOURCE_BARRIER::Transition(
            m_vertexBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
        );
        
        m_commandList->ResourceBarrier(1, &barrierBack);
        
        const char* debugMsg = "ResourceBarrier transition completed";
        // Debug output - this literal should not be counted
    }
    
    // Process index buffer for indexed rendering
    void ProcessIndexBuffer(ComPtr<ID3D12Resource> indexBuffer, UINT indexCount) {
        m_indexBuffer = indexBuffer;
        
        // Transition index buffer for copy operation
        // Note: We use ResourceBarrier here for legacy compatibility
        CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_indexBuffer.Get(),
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            D3D12_RESOURCE_STATE_COPY_DEST
        );
        
        m_commandList->ResourceBarrier(1, &barrier);
        
        // Perform index buffer operations here
        // ... actual copy operations would go here ...
        
        // Transition back to index buffer state
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_indexBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDEX_BUFFER
            )
        );
    }
    
    // Batch process multiple geometry buffers
    void BatchProcessGeometry(
        ComPtr<ID3D12Resource>* buffers,
        UINT bufferCount,
        D3D12_RESOURCE_STATES initialState,
        D3D12_RESOURCE_STATES targetState
    ) {
        // Allocate barrier array for batch processing
        CD3DX12_RESOURCE_BARRIER* barriers = new CD3DX12_RESOURCE_BARRIER[bufferCount];
        
        for (UINT i = 0; i < bufferCount; i++) {
            // Create transition barriers for each buffer
            barriers[i] = CD3DX12_RESOURCE_BARRIER::Transition(
                buffers[i].Get(),
                initialState,
                targetState
            );
        }
        
        // Submit all barriers in a single call
        // Legacy API: ResourceBarrier with multiple barriers
        m_commandList->ResourceBarrier(bufferCount, barriers);
        
        delete[] barriers;
    }
    
    // Modern barrier implementation using Enhanced Barriers API
    void OptimizedGeometryTransition(ComPtr<ID3D12Resource> resource) {
        if (!m_commandList7) {
            throw std::runtime_error("Enhanced Barriers API not available");
        }
        
        // Enhanced Barriers API usage
        // This uses the new D3D12_BARRIER_GROUP structure
        D3D12_BUFFER_BARRIER bufferBarrier = {};
        bufferBarrier.SyncBefore = D3D12_BARRIER_SYNC_COPY;
        bufferBarrier.SyncAfter = D3D12_BARRIER_SYNC_VERTEX_SHADING;
        bufferBarrier.AccessBefore = D3D12_BARRIER_ACCESS_COPY_DEST;
        bufferBarrier.AccessAfter = D3D12_BARRIER_ACCESS_VERTEX_BUFFER;
        bufferBarrier.pResource = resource.Get();
        bufferBarrier.Offset = 0;
        bufferBarrier.Size = UINT64_MAX;
        
        D3D12_BARRIER_GROUP barrierGroup = {};
        barrierGroup.Type = D3D12_BARRIER_TYPE_BUFFER;
        barrierGroup.NumBarriers = 1;
        barrierGroup.pBufferBarriers = &bufferBarrier;
        
        // Enhanced barrier API: Barrier method call
        m_commandList7->Barrier(1, &barrierGroup);
    }
    
    // Synchronize geometry processing pipeline stages
    void SynchronizeGeometryStages() {
        // Use UAV barrier for geometry shader output synchronization
        // Legacy API pattern
        auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_vertexBuffer.Get());
        m_commandList->ResourceBarrier(1, &uavBarrier);
    }
    
    // Prepare geometry buffers for rendering
    void PrepareForRendering(
        ComPtr<ID3D12Resource> vertexBuffer,
        ComPtr<ID3D12Resource> indexBuffer
    ) {
        // Transition vertex buffer to appropriate state
        // Legacy barrier usage
        D3D12_RESOURCE_BARRIER barriers[2] = {};
        
        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            vertexBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
        );
        
        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            indexBuffer.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_INDEX_BUFFER
        );
        
        // Submit both barriers together
        // This is a legacy ResourceBarrier call
        m_commandList->ResourceBarrier(2, barriers);
    }
};

// Global helper function for geometry resource initialization
void InitializeGeometryResources(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES initialState
) {
    // Transition from initial state to common state
    // Legacy API usage for initialization
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        resource,
        initialState,
        D3D12_RESOURCE_STATE_COMMON
    );
    
    commandList->ResourceBarrier(1, &barrier);
}