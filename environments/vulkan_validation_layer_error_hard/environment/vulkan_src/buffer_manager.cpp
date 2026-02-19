#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <cstring>

class BufferManager {
private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        
        throw std::runtime_error("Failed to find suitable memory type");
    }

public:
    BufferManager(VkDevice dev, VkPhysicalDevice physDev) 
        : device(dev), physicalDevice(physDev) {}
    
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                     VkMemoryPropertyFlags properties, VkBuffer& buffer, 
                     VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.flags = 0;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory");
        }
        
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    
    void createVertexBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        createBuffer(size, 
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    buffer, bufferMemory);
    }
    
    void createStagingBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        createBuffer(size,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    buffer, bufferMemory);
    }
    
    void createUniformBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        createBuffer(size,
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    buffer, bufferMemory);
    }
    
    void copyDataToBuffer(VkDeviceMemory bufferMemory, const void* srcData, VkDeviceSize size) {
        void* data;
        vkMapMemory(device, bufferMemory, 0, size, 0, &data);
        memcpy(data, srcData, static_cast<size_t>(size));
        vkUnmapMemory(device, bufferMemory);
    }
    
    void destroyBuffer(VkBuffer buffer, VkDeviceMemory bufferMemory) {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
    }
    
    void createIndexBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        createBuffer(size,
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    buffer, bufferMemory);
    }
};