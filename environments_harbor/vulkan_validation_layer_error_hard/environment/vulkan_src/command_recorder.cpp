// command_recorder.cpp
// Vulkan command buffer recorder with various synchronization operations

#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>

class CommandRecorder {
private:
    VkDevice device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    
public:
    CommandRecorder(VkDevice dev, VkCommandPool pool) 
        : device(dev), commandPool(pool), commandBuffer(VK_NULL_HANDLE) {}
    
    void createCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }
    }
    
    // Records buffer update and vertex binding without proper barrier
    void recordBufferUpdateAndDraw(VkBuffer stagingBuffer, VkBuffer vertexBuffer, 
                                   VkDeviceSize size, VkPipeline pipeline) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        // Copy from staging to vertex buffer
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, stagingBuffer, vertexBuffer, 1, &copyRegion);
        
        // VIOLATION: Missing memory barrier here
        // Should have VkBufferMemoryBarrier with srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
        // and dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
        
        // Immediately bind the vertex buffer without synchronization
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        
        vkEndCommandBuffer(commandBuffer);
    }
    
    // Records image operations with missing layout transition barriers
    void recordImageOperations(VkImage srcImage, VkImage dstImage, 
                               VkExtent2D extent) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        // Copy image without proper layout transitions
        VkImageCopy copyRegion{};
        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.srcSubresource.mipLevel = 0;
        copyRegion.srcSubresource.baseArrayLayer = 0;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.dstSubresource = copyRegion.srcSubresource;
        copyRegion.srcOffset = {0, 0, 0};
        copyRegion.dstOffset = {0, 0, 0};
        copyRegion.extent = {extent.width, extent.height, 1};
        
        // VIOLATION: Assuming images are in correct layout without barriers
        // Should transition srcImage to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        // and dstImage to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL first
        vkCmdCopyImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      1, &copyRegion);
        
        vkEndCommandBuffer(commandBuffer);
    }
    
    // Records buffer updates with missing execution dependencies
    void recordMultipleBufferUpdates(VkBuffer uniformBuffer, VkBuffer storageBuffer,
                                     const void* uniformData, VkDeviceSize uniformSize) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        // Update uniform buffer
        vkCmdUpdateBuffer(commandBuffer, uniformBuffer, 0, uniformSize, uniformData);
        
        // VIOLATION: Missing pipeline barrier between update and usage
        // The storage buffer operation depends on uniform buffer write completing
        
        // Fill storage buffer - assumes uniform buffer write is complete
        vkCmdFillBuffer(commandBuffer, storageBuffer, 0, VK_WHOLE_SIZE, 0);
        
        vkEndCommandBuffer(commandBuffer);
    }
    
    // Records compute dispatch with missing memory barriers
    void recordComputeOperations(VkPipeline computePipeline, VkBuffer dataBuffer,
                                 VkDescriptorSet descriptorSet) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               VK_NULL_HANDLE, 0, 1, &descriptorSet, 0, nullptr);
        
        // First compute dispatch
        vkCmdDispatch(commandBuffer, 256, 1, 1);
        
        // VIOLATION: Missing memory barrier between dispatches
        // Second dispatch reads what first dispatch wrote
        // Should have VkBufferMemoryBarrier with srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT
        // and dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        
        // Second compute dispatch immediately follows
        vkCmdDispatch(commandBuffer, 256, 1, 1);
        
        vkEndCommandBuffer(commandBuffer);
    }
    
    // Records render pass with image layout issues
    void recordRenderPass(VkRenderPass renderPass, VkFramebuffer framebuffer,
                         VkExtent2D extent, VkImage colorImage) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffer;
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = extent;
        
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
        
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        // Render pass operations here
        
        vkCmdEndRenderPass(commandBuffer);
        
        // VIOLATION: After render pass, image is in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        // but no barrier to transition to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR or other usage
        
        vkEndCommandBuffer(commandBuffer);
    }
    
    VkCommandBuffer getCommandBuffer() const {
        return commandBuffer;
    }
    
    void cleanup() {
        if (commandBuffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            commandBuffer = VK_NULL_HANDLE;
        }
    }
};