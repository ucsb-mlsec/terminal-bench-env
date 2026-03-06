// Vulkan Resource Management - Command Buffer Recording
// This file handles cleanup and resource management for Vulkan objects

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>

class ResourceManager {
private:
    VkDevice device;
    VkCommandPool commandPool;
    VkQueue graphicsQueue;
    
public:
    ResourceManager(VkDevice dev, VkCommandPool pool, VkQueue queue) 
        : device(dev), commandPool(pool), graphicsQueue(queue) {}
    
    void cleanupAndRender() {
        // Create some resources for rendering
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkImage textureImage;
        VkDeviceMemory textureImageMemory;
        VkImageView textureImageView;
        VkPipeline graphicsPipeline;
        VkDescriptorSet descriptorSet;
        
        // Create vertex buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = 65536;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer);
        
        // Allocate memory for vertex buffer
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = 65536;
        allocInfo.memoryTypeIndex = 0;
        vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory);
        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
        
        // Create texture image
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = 512;
        imageInfo.extent.height = 512;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateImage(device, &imageInfo, nullptr, &textureImage);
        
        // Allocate and bind image memory
        vkAllocateMemory(device, &allocInfo, nullptr, &textureImageMemory);
        vkBindImageMemory(device, textureImage, textureImageMemory, 0);
        
        // Create image view
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = textureImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &viewInfo, nullptr, &textureImageView);
        
        // VIOLATION: Destroy the buffer early but still use it later
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        
        // Create command buffer
        VkCommandBuffer commandBuffer;
        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        // VIOLATION: Trying to bind a destroyed vertex buffer
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
        
        // VIOLATION: Destroy the image view before using it in descriptor set
        vkDestroyImageView(device, textureImageView, nullptr);
        
        // Attempt to update descriptor set with destroyed image view (simulated)
        VkDescriptorImageInfo imageDescriptor{};
        imageDescriptor.imageView = textureImageView; // Using destroyed view
        imageDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        
        // VIOLATION: Free memory while resources are still bound
        vkFreeMemory(device, textureImageMemory, nullptr);
        
        // Continue using the image even though its memory is freed
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image = textureImage;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        
        vkCmdPipelineBarrier(commandBuffer, 
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
        
        // VIOLATION: Destroy pipeline then try to bind it
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        
        vkEndCommandBuffer(commandBuffer);
        
        // Submit command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        
        // VIOLATION: Free vertex buffer memory while it might still be in use
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        
        // Cleanup remaining resources
        vkDestroyImage(device, textureImage, nullptr);
    }
};