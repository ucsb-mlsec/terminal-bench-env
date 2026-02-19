#include <d3d12.h>
#include "d3dx12.h"
#include "compute_pipeline.h"
#include <vector>
#include <cassert>

namespace Renderer {
namespace Compute {

// ComputePipeline constructor
ComputePipeline::ComputePipeline(ID3D12Device* device)
    : m_device(device)
    , m_commandList(nullptr)
    , m_pipelineState(nullptr)
    , m_rootSignature(nullptr)
{
    assert(m_device != nullptr);
}

// Initialize compute pipeline with shader bytecode
bool ComputePipeline::Initialize(const void* shaderBytecode, size_t bytecodeLength)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature;
    psoDesc.CS.pShaderBytecode = shaderBytecode;
    psoDesc.CS.BytecodeLength = bytecodeLength;

    HRESULT hr = m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState));
    return SUCCEEDED(hr);
}

// Dispatch compute shader with UAV resource synchronization
void ComputePipeline::DispatchCompute(ID3D12GraphicsCommandList* commandList,
                                      ID3D12Resource* inputResource,
                                      ID3D12Resource* outputResource,
                                      UINT threadGroupX,
                                      UINT threadGroupY,
                                      UINT threadGroupZ)
{
    assert(commandList != nullptr);
    assert(inputResource != nullptr);
    assert(outputResource != nullptr);

    // Transition input resource to shader resource state
    D3D12_RESOURCE_BARRIER barriers[2];
    barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
        inputResource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Transition output resource to UAV state
    barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
        outputResource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    commandList->ResourceBarrier(2, barriers);

    // Set pipeline state and dispatch
    commandList->SetPipelineState(m_pipelineState);
    commandList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);

    // Insert UAV barrier to ensure compute shader writes are complete
    D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(outputResource);
    commandList->ResourceBarrier(1, &uavBarrier);
}

// Execute multiple compute passes with UAV barriers between passes
void ComputePipeline::ExecuteMultiPass(ID3D12GraphicsCommandList* commandList,
                                       ID3D12Resource* workBuffer,
                                       const std::vector<ComputePass>& passes)
{
    assert(commandList != nullptr);
    assert(workBuffer != nullptr);

    // Transition work buffer to UAV state for compute operations
    D3D12_RESOURCE_BARRIER initialBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        workBuffer,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    
    commandList->ResourceBarrier(1, &initialBarrier);

    for (size_t i = 0; i < passes.size(); ++i)
    {
        const ComputePass& pass = passes[i];
        
        // Set compute root signature and pipeline state
        commandList->SetComputeRootSignature(pass.rootSignature);
        commandList->SetPipelineState(pass.pipelineState);
        commandList->SetComputeRootDescriptorTable(0, pass.srvHandle);
        commandList->SetComputeRootDescriptorTable(1, pass.uavHandle);

        // Dispatch compute work
        commandList->Dispatch(pass.threadGroupX, pass.threadGroupY, pass.threadGroupZ);

        // Insert UAV barrier between passes to ensure data dependencies are satisfied
        if (i < passes.size() - 1)
        {
            D3D12_RESOURCE_BARRIER passBarrier = CD3DX12_RESOURCE_BARRIER::UAV(workBuffer);
            commandList->ResourceBarrier(1, &passBarrier);
        }
    }

    // Transition work buffer back to common state after all passes complete
    D3D12_RESOURCE_BARRIER finalBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        workBuffer,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON);
    
    commandList->ResourceBarrier(1, &finalBarrier);
}

// Perform reduction operation with intermediate UAV synchronization
void ComputePipeline::PerformReduction(ID3D12GraphicsCommandList* commandList,
                                       ID3D12Resource* sourceBuffer,
                                       ID3D12Resource* intermediateBuffer,
                                       ID3D12Resource* resultBuffer)
{
    assert(commandList != nullptr);

    // Prepare source buffer for reading
    D3D12_RESOURCE_BARRIER preBarriers[2];
    preBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
        sourceBuffer,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    preBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
        intermediateBuffer,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    commandList->ResourceBarrier(2, preBarriers);

    // First reduction pass
    commandList->SetPipelineState(m_pipelineState);
    commandList->Dispatch(256, 1, 1);

    // Synchronize intermediate buffer writes
    D3D12_RESOURCE_BARRIER intermediateUavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(intermediateBuffer);
    commandList->ResourceBarrier(1, &intermediateUavBarrier);

    // Second reduction pass
    commandList->Dispatch(16, 1, 1);

    // Final UAV barrier before reading result
    D3D12_RESOURCE_BARRIER finalUavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(resultBuffer);
    commandList->ResourceBarrier(1, &finalUavBarrier);
}

// Cleanup resources
void ComputePipeline::Shutdown()
{
    if (m_pipelineState)
    {
        m_pipelineState->Release();
        m_pipelineState = nullptr;
    }

    if (m_rootSignature)
    {
        m_rootSignature->Release();
        m_rootSignature = nullptr;
    }
}

} // namespace Compute
} // namespace Renderer