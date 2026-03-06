#include <vulkan/vulkan.h>
#include <stdexcept>
#include <cstring>

class VulkanMemoryAllocator {
private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceMemoryProperties memoryProperties;

public:
    VulkanMemoryAllocator(VkDevice dev, VkPhysicalDevice physDev) 
        : device(dev), physicalDevice(physDev) {
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }

    VkDeviceMemory allocateBufferMemory(VkBuffer buffer, VkMemoryPropertyFlags properties) {
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        VkDeviceMemory memory;
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory");
        }

        return memory;
    }

    VkDeviceMemory allocateImageMemory(VkImage image, VkMemoryPropertyFlags properties) {
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        VkDeviceMemory memory;
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate image memory");
        }

        return memory;
    }

    void copyDataToMemory(VkDeviceMemory memory, const void* data, VkDeviceSize size, VkDeviceSize offset = 0) {
        void* mappedData;
        vkMapMemory(device, memory, offset, size, 0, &mappedData);
        memcpy(mappedData, data, static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }

    void copyDataFromMemory(VkDeviceMemory memory, void* data, VkDeviceSize size, VkDeviceSize offset = 0) {
        void* mappedData;
        vkMapMemory(device, memory, offset, size, 0, &mappedData);
        memcpy(data, mappedData, static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }

    bool isMemoryTypeDeviceLocal(uint32_t typeIndex) {
        return (memoryProperties.memoryTypes[typeIndex].propertyFlags & 
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    }

    bool isMemoryTypeHostVisible(uint32_t typeIndex) {
        return (memoryProperties.memoryTypes[typeIndex].propertyFlags & 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    bool isMemoryTypeHostCoherent(uint32_t typeIndex) {
        return (memoryProperties.memoryTypes[typeIndex].propertyFlags & 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }

    VkDeviceSize getMemoryHeapSize(uint32_t typeIndex) {
        uint32_t heapIndex = memoryProperties.memoryTypes[typeIndex].heapIndex;
        return memoryProperties.memoryHeaps[heapIndex].size;
    }

    void freeMemory(VkDeviceMemory memory) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
        }
    }

    VkDeviceMemory allocateMemoryWithRequirements(VkMemoryRequirements requirements, 
                                                   VkMemoryPropertyFlags properties) {
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = requirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);

        VkDeviceMemory memory;
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate device memory");
        }

        return memory;
    }
};