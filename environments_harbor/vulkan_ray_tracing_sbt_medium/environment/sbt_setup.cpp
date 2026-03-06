#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include "shader_config.h"

// Global device properties (would normally be retrieved during device initialization)
struct RayTracingProperties {
    uint32_t shaderGroupHandleSize;
    uint32_t shaderGroupHandleAlignment;
    uint32_t shaderGroupBaseAlignment;
    uint32_t maxShaderGroupStride;
} rtProperties;

// SBT buffer handle
VkBuffer sbtBuffer = VK_NULL_HANDLE;
VkDeviceMemory sbtMemory = VK_NULL_HANDLE;
VkDeviceAddress sbtAddress = 0;

// Strided device address regions for each shader group type
VkStridedDeviceAddressRegionKHR raygenRegion = {};
VkStridedDeviceAddressRegionKHR missRegion = {};
VkStridedDeviceAddressRegionKHR hitRegion = {};
VkStridedDeviceAddressRegionKHR callableRegion = {};

// Helper function to align values (provided but not used correctly)
uint32_t alignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Function to setup the Shader Binding Table with BUGGY alignment
void setupShaderBindingTable(VkDevice device, VkPipeline rayTracingPipeline) {
    // Get the handle size from device properties
    uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;
    
    // Calculate the number of shader groups from configuration
    uint32_t raygenGroupCount = RAYGEN_SHADER_COUNT;  // From shader_config.h
    uint32_t missGroupCount = MISS_SHADER_COUNT;      // From shader_config.h
    uint32_t hitGroupCount = HIT_SHADER_COUNT;        // From shader_config.h
    uint32_t callableGroupCount = CALLABLE_SHADER_COUNT; // From shader_config.h
    
    uint32_t totalGroupCount = raygenGroupCount + missGroupCount + hitGroupCount + callableGroupCount;
    
    // Get all shader group handles from the pipeline
    std::vector<uint8_t> shaderHandleStorage(totalGroupCount * handleSize);
    vkGetRayTracingShaderGroupHandlesKHR(device, rayTracingPipeline, 0, totalGroupCount,
                                         totalGroupCount * handleSize, shaderHandleStorage.data());
    
    // BUGGY STRIDE CALCULATIONS - Not properly aligned!
    // According to Vulkan spec, strides must be aligned to shaderGroupHandleAlignment
    // and the base offset must be aligned to shaderGroupBaseAlignment
    
    // BUG: Ray generation stride should be aligned but we're just using handleSize
    uint32_t raygenStride = handleSize;  // WRONG! Missing alignment
    
    // BUG: Miss shader stride calculation is also wrong
    uint32_t missStride = handleSize;  // WRONG! Should be aligned to handleAlignment
    
    // BUG: Hit shader has additional data (4 bytes) but not properly aligned
    // Hit groups might have shader records with additional data
    uint32_t hitStride = handleSize + 4;  // WRONG! Arbitrary addition without proper alignment
    
    // BUG: Callable shader stride is also incorrect
    uint32_t callableStride = handleSize;  // WRONG! Missing alignment requirements
    
    // Calculate sizes for each SBT region
    // These calculations use the buggy strides above
    uint32_t raygenSize = raygenGroupCount * raygenStride;
    uint32_t missSize = missGroupCount * missStride;
    uint32_t hitSize = hitGroupCount * hitStride;
    uint32_t callableSize = callableGroupCount * callableStride;
    
    // Calculate total SBT buffer size
    // BUG: Not accounting for base alignment between regions
    uint32_t totalSize = raygenSize + missSize + hitSize + callableSize;
    
    // Create SBT buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = totalSize;
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | 
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    
    vkCreateBuffer(device, &bufferInfo, nullptr, &sbtBuffer);
    
    // Allocate memory for SBT (simplified - real code would query requirements)
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = totalSize;
    
    // Enable device address flag
    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocInfo.pNext = &flagsInfo;
    
    vkAllocateMemory(device, &allocInfo, nullptr, &sbtMemory);
    vkBindBufferMemory(device, sbtBuffer, sbtMemory, 0);
    
    // Get buffer device address
    VkBufferDeviceAddressInfo addressInfo = {};
    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addressInfo.buffer = sbtBuffer;
    sbtAddress = vkGetBufferDeviceAddress(device, &addressInfo);
    
    // Map memory and copy shader handles
    void* mappedData;
    vkMapMemory(device, sbtMemory, 0, totalSize, 0, &mappedData);
    
    uint8_t* pData = reinterpret_cast<uint8_t*>(mappedData);
    uint32_t handleIndex = 0;
    
    // Copy ray generation handles
    // BUG: Using incorrect stride for copying
    for (uint32_t i = 0; i < raygenGroupCount; i++) {
        memcpy(pData, shaderHandleStorage.data() + handleIndex * handleSize, handleSize);
        pData += raygenStride;
        handleIndex++;
    }
    
    // Copy miss shader handles
    // BUG: Using incorrect stride
    for (uint32_t i = 0; i < missGroupCount; i++) {
        memcpy(pData, shaderHandleStorage.data() + handleIndex * handleSize, handleSize);
        pData += missStride;
        handleIndex++;
    }
    
    // Copy hit group handles
    // BUG: Using incorrect stride with arbitrary additional data
    for (uint32_t i = 0; i < hitGroupCount; i++) {
        memcpy(pData, shaderHandleStorage.data() + handleIndex * handleSize, handleSize);
        // The +4 bytes are for shader record data, but stride is still wrong
        pData += hitStride;
        handleIndex++;
    }
    
    // Copy callable shader handles
    // BUG: Using incorrect stride
    for (uint32_t i = 0; i < callableGroupCount; i++) {
        memcpy(pData, shaderHandleStorage.data() + handleIndex * handleSize, handleSize);
        pData += callableStride;
        handleIndex++;
    }
    
    vkUnmapMemory(device, sbtMemory);
    
    // Setup strided device address regions
    // BUG: These regions use the incorrectly calculated strides and sizes
    
    VkDeviceAddress currentAddress = sbtAddress;
    
    // Ray generation region
    raygenRegion.deviceAddress = currentAddress;
    raygenRegion.stride = raygenStride;  // WRONG stride
    raygenRegion.size = raygenSize;
    currentAddress += raygenSize;
    
    // Miss shader region
    missRegion.deviceAddress = currentAddress;
    missRegion.stride = missStride;  // WRONG stride
    missRegion.size = missSize;
    currentAddress += missSize;
    
    // Hit group region
    hitRegion.deviceAddress = currentAddress;
    hitRegion.stride = hitStride;  // WRONG stride
    hitRegion.size = hitSize;
    currentAddress += hitSize;
    
    // Callable shader region
    callableRegion.deviceAddress = currentAddress;
    callableRegion.stride = callableStride;  // WRONG stride
    callableRegion.size = callableSize;
    
    // NOTE: The Vulkan specification requires:
    // 1. stride must be a multiple of shaderGroupHandleAlignment
    // 2. stride must be less than or equal to maxShaderGroupStride
    // 3. size must be a multiple of stride (for non-raygen)
    // 4. The base address of each region should respect shaderGroupBaseAlignment
    // 
    // Our current implementation violates these requirements!
}

// Function to trace rays using the SBT (would be called in render loop)
void traceRays(VkCommandBuffer commandBuffer, uint32_t width, uint32_t height) {
    vkCmdTraceRaysKHR(commandBuffer,
                      &raygenRegion,
                      &missRegion,
                      &hitRegion,
                      &callableRegion,
                      width, height, 1);
}

// Cleanup function
void cleanupSBT(VkDevice device) {
    if (sbtBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, sbtBuffer, nullptr);
    }
    if (sbtMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, sbtMemory, nullptr);
    }
}