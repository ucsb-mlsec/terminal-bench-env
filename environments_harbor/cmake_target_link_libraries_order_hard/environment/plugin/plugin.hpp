#pragma once

#include "../core/core.hpp"

namespace Plugin {

class PluginImpl : public Core::CoreBase {
public:
    PluginImpl();
    virtual ~PluginImpl();
    
    void execute() override;
    
    void pluginFeature();
};

} // namespace Plugin