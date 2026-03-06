#include <iostream>
#include <memory>
#include "core/database.h"
#include "api/handler.h"

int main() {
    std::cout << "Initializing application..." << std::endl;
    
    // Initialize database connection
    auto db = std::make_shared<Database>();
    if (!db->connect()) {
        std::cerr << "Failed to connect to database" << std::endl;
        return 1;
    }
    
    // Setup API handler
    ApiHandler handler(db);
    handler.initialize();
    
    std::cout << "Application started successfully" << std::endl;
    
    // Main event loop
    handler.run();
    
    // Cleanup
    db->disconnect();
    std::cout << "Application shutdown complete" << std::endl;
    
    return 0;
}