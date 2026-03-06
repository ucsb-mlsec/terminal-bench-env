#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>

class ShaderLoader {
private:
    VkDevice device;

    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open shader file: " + filename);
        }
        
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        
        return buffer;
    }

public:
    ShaderLoader(VkDevice dev) : device(dev) {}

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        createInfo.pNext = nullptr;
        createInfo.flags = 0;

        VkShaderModule shaderModule;
        VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
        
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module");
        }

        return shaderModule;
    }

    VkShaderModule loadShaderFromFile(const std::string& filename) {
        std::vector<char> shaderCode = readFile(filename);
        return createShaderModule(shaderCode);
    }

    VkPipelineShaderStageCreateInfo createVertexShaderStage(VkShaderModule shaderModule) {
        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";
        shaderStageInfo.pSpecializationInfo = nullptr;
        shaderStageInfo.flags = 0;
        shaderStageInfo.pNext = nullptr;

        return shaderStageInfo;
    }

    VkPipelineShaderStageCreateInfo createFragmentShaderStage(VkShaderModule shaderModule) {
        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";
        shaderStageInfo.pSpecializationInfo = nullptr;
        shaderStageInfo.flags = 0;
        shaderStageInfo.pNext = nullptr;

        return shaderStageInfo;
    }

    VkPipelineShaderStageCreateInfo createShaderStageWithSpecialization(
        VkShaderModule shaderModule,
        VkShaderStageFlagBits stage,
        const VkSpecializationInfo* specializationInfo) {
        
        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = stage;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";
        shaderStageInfo.pSpecializationInfo = specializationInfo;
        shaderStageInfo.flags = 0;
        shaderStageInfo.pNext = nullptr;

        return shaderStageInfo;
    }

    void destroyShaderModule(VkShaderModule shaderModule) {
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }

    void loadAndCreateShaderStages(const std::string& vertPath, const std::string& fragPath,
                                   VkShaderModule& vertModule, VkShaderModule& fragModule,
                                   VkPipelineShaderStageCreateInfo* stages) {
        vertModule = loadShaderFromFile(vertPath);
        fragModule = loadShaderFromFile(fragPath);

        stages[0] = createVertexShaderStage(vertModule);
        stages[1] = createFragmentShaderStage(fragModule);
    }
};