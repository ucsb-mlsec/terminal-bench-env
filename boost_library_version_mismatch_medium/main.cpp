#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <string>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

int main(int argc, char* argv[]) {
    try {
        ptree pt;
        
        // Read the JSON file using the old Boost 1.58 style
        // In Boost 1.58, read_json accepted std::string directly
        std::string filename = "sample.json";
        read_json(filename, pt);
        
        std::cout << "Successfully loaded JSON configuration file" << std::endl;
        
        // Try to read various configuration values
        // Using the old-style get method that may have changed
        std::string serverHost = pt.get<std::string>("server.host");
        int serverPort = pt.get<int>("server.port");
        std::string dbName = pt.get<std::string>("database.name");
        
        std::cout << "Server Host: " << serverHost << std::endl;
        std::cout << "Server Port: " << serverPort << std::endl;
        std::cout << "Database Name: " << dbName << std::endl;
        
        // Try to access nested configuration
        std::string logLevel = pt.get<std::string>("logging.level", "INFO");
        std::cout << "Log Level: " << logLevel << std::endl;
        
        // Access array-like structure using old patterns
        ptree& features = pt.get_child("features");
        std::cout << "Features enabled:" << std::endl;
        for (ptree::iterator it = features.begin(); it != features.end(); ++it) {
            std::cout << "  - " << it->second.get_value<std::string>() << std::endl;
        }
        
        std::cout << "Configuration loaded successfully!" << std::endl;
        
    } catch (const boost::property_tree::ptree_error& e) {
        std::cerr << "Property tree error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}