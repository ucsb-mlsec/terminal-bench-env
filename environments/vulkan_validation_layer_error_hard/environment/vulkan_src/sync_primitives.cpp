#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>

class SyncPrimitives {
private:
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkFence renderFence;
    VkFence computeFence;
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkCommandBuffer commandBuffer;
    VkCommandBuffer computeCommandBuffer;

public:
    SyncPrimitives(VkDevice dev, VkQueue gfxQueue, VkQueue compQueue) 
        : device(dev), graphicsQueue(gfxQueue), computeQueue(compQueue) {
        createSyncObjects();
    }

    void createSyncObjects() {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        vkCreateFence(device, &fenceInfo, nullptr, &renderFence);
        vkCreateFence(device, &fenceInfo, nullptr, &computeFence);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore);
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore);
    }

    void executeRenderPass() {
        // VIOLATION: Waiting for fence that was never submitted
        vkWaitForFences(device, 1, &renderFence, VK_TRUE, UINT64_MAX);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
        // VIOLATION: waitStageMask doesn't match actual usage - using TOP_OF_PIPE for color attachment output
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphore;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, renderFence);
    }

    void executeComputePass() {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &computeCommandBuffer;

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &renderFinishedSemaphore;
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
        submitInfo.pWaitDstStageMask = waitStages;

        // VIOLATION: Signaling the same semaphore that another operation might signal
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphore;

        vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence);
    }

    void submitMultipleFrames() {
        for (int i = 0; i < 3; i++) {
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &imageAvailableSemaphore;

            // VIOLATION: Reusing fence without calling vkResetFences
            vkQueueSubmit(graphicsQueue, 1, &submitInfo, renderFence);
        }
    }

    void dependentSubmissions() {
        VkSubmitInfo firstSubmit{};
        firstSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        firstSubmit.commandBufferCount = 1;
        firstSubmit.pCommandBuffers = &commandBuffer;

        // VIOLATION: No synchronization between dependent submissions
        vkQueueSubmit(graphicsQueue, 1, &firstSubmit, VK_NULL_HANDLE);

        VkSubmitInfo secondSubmit{};
        secondSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        secondSubmit.commandBufferCount = 1;
        secondSubmit.pCommandBuffers = &computeCommandBuffer;
        
        // This depends on first submission but has no semaphore wait
        vkQueueSubmit(graphicsQueue, 1, &secondSubmit, VK_NULL_HANDLE);
    }

    void simultaneousSignaling() {
        VkSubmitInfo submit1{};
        submit1.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit1.commandBufferCount = 1;
        submit1.pCommandBuffers = &commandBuffer;
        submit1.signalSemaphoreCount = 1;
        submit1.pSignalSemaphores = &imageAvailableSemaphore;

        VkSubmitInfo submit2{};
        submit2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit2.commandBufferCount = 1;
        submit2.pCommandBuffers = &computeCommandBuffer;
        submit2.signalSemaphoreCount = 1;
        // VIOLATION: Signaling same semaphore in multiple submissions without wait
        submit2.pSignalSemaphores = &imageAvailableSemaphore;

        vkQueueSubmit(graphicsQueue, 1, &submit1, VK_NULL_HANDLE);
        vkQueueSubmit(computeQueue, 1, &submit2, VK_NULL_HANDLE);
    }

    ~SyncPrimitives() {
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroyFence(device, renderFence, nullptr);
        vkDestroyFence(device, computeFence, nullptr);
    }
};