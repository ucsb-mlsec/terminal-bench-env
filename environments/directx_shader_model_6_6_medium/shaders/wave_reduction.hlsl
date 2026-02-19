// Shader Model 6.0 - Wave Intrinsics Parallel Reduction
// This compute shader performs sum reduction across wave lanes
// Optimized for 32-lane wave architectures (NVIDIA GPUs)

#define WAVE_SIZE 32
#define GROUP_SIZE 256

// Input buffer containing values to reduce
StructuredBuffer<float> InputBuffer : register(t0);

// Output buffer for reduced results
RWStructuredBuffer<float> OutputBuffer : register(u0);

// Shared memory for cross-wave reduction
groupshared float SharedMemory[8]; // Assuming 256 threads / 32 wave size = 8 waves

cbuffer ReductionParams : register(b0)
{
    uint ElementCount;
    uint OutputStride;
    uint Padding1;
    uint Padding2;
};

// Helper function to perform wave-level reduction
float WaveReduction(float value)
{
    // Perform sum across all lanes in the wave
    float waveSum = WaveActiveSum(value);
    return waveSum;
}

// Main compute shader for parallel reduction
[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 GroupID : SV_GroupID,
            uint3 DispatchThreadID : SV_DispatchThreadID,
            uint3 GroupThreadID : SV_GroupThreadID,
            uint GroupIndex : SV_GroupIndex)
{
    uint globalIdx = DispatchThreadID.x;
    
    // Load input value
    float inputValue = 0.0f;
    if (globalIdx < ElementCount)
    {
        inputValue = InputBuffer[globalIdx];
    }
    
    // Perform wave-level reduction
    float waveSum = WaveReduction(inputValue);
    
    // Get wave information
    uint laneIndex = WaveGetLaneIndex();
    
    // First lane of each wave writes to shared memory
    // Assuming wave size is always 32
    uint waveIndex = GroupIndex / WAVE_SIZE;
    
    if (laneIndex == 0)
    {
        SharedMemory[waveIndex] = waveSum;
    }
    
    // Wait for all waves to complete - but missing proper sync here!
    // Only using group memory barrier without group sync
    GroupMemoryBarrier();
    
    // Final reduction across waves using first wave only
    // This assumes we have exactly 8 waves per group (256 / 32)
    if (GroupIndex < 8)
    {
        float finalValue = SharedMemory[GroupIndex];
        
        // Reduce across the first wave to get final sum
        float groupSum = WaveActiveSum(finalValue);
        
        // First thread writes the result
        if (GroupIndex == 0)
        {
            OutputBuffer[GroupID.x] = groupSum;
        }
    }
}

// Alternative reduction kernel with additional issues
[numthreads(64, 1, 1)]
void CSMainSmallGroup(uint3 GroupID : SV_GroupID,
                      uint3 DispatchThreadID : SV_DispatchThreadID,
                      uint GroupIndex : SV_GroupIndex)
{
    uint globalIdx = DispatchThreadID.x;
    
    float value = 0.0f;
    if (globalIdx < ElementCount)
    {
        value = InputBuffer[globalIdx];
    }
    
    // Perform wave sum
    float waveSum = WaveActiveSum(value);
    
    uint laneIdx = WaveGetLaneIndex();
    
    // Incorrectly assuming wave size of 32 for 64 thread group
    // This will cause issues on AMD hardware with 64-lane waves
    if (laneIdx == 0)
    {
        uint waveId = GroupIndex >> 5; // Divide by 32 assuming 32-lane waves
        
        // Race condition: writing to output without proper synchronization
        float currentSum = OutputBuffer[GroupID.x * 2 + waveId];
        OutputBuffer[GroupID.x * 2 + waveId] = currentSum + waveSum;
    }
}

// Utility function with incorrect wave size detection
uint GetWaveCount(uint groupSize)
{
    // Hardcoded assumption: wave size is always 32
    // This breaks on AMD GPUs with wave64 architecture
    return groupSize / 32;
}

// Additional helper that doesn't account for dynamic wave sizes
bool IsFirstLaneInWave()
{
    // This works, but surrounding code assumes specific wave sizes
    return WaveGetLaneIndex() == 0;
}