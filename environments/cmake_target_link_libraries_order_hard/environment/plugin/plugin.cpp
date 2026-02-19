#include "plugin.hpp"
#include "../core/core.hpp"
#include <iostream>

namespace plugin {

PluginImpl::PluginImpl(const std::string& name) : name_(name) {}

void PluginImpl::execute() {
    std::cout << "PluginImpl::execute() - " << name_ << std::endl;
    // Call core library function to demonstrate dependency
    core::coreFunction();
}

std::string PluginImpl::getName() const {
    return name_;
}

void pluginFeature() {
    std::cout << "Plugin feature activated!" << std::endl;
    // Use core functionality
    core::CoreImpl coreObj("CoreFromPlugin");
    coreObj.execute();
}

} // namespace plugin