#include <iostream>
#include "../core/core.hpp"
#include "../plugin/plugin.hpp"

int main() {
    std::cout << "Plugin System Application" << std::endl;
    
    // Call core library function
    std::cout << "Core Version: " << getCoreVersion() << std::endl;
    
    // Create core object
    CoreBase* coreObj = new CoreBase();
    std::cout << "Core Message: " << coreObj->getMessage() << std::endl;
    
    // Create plugin object (extends CoreBase)
    Plugin* pluginObj = new Plugin();
    std::cout << "Plugin Message: " << pluginObj->getMessage() << std::endl;
    std::cout << "Plugin Feature: " << pluginObj->getPluginFeature() << std::endl;
    
    // Cleanup
    delete coreObj;
    delete pluginObj;
    
    std::cout << "Application completed successfully!" << std::endl;
    
    return 0;
}