#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using json = nlohmann::json;

int main() {
    try {
        // Open the config file
        ifstream configFile("../config.json");
        if (!configFile.is_open()) {
            cerr << "Error: Could not open config.json" << endl;
            return 1;
        }

        // Parse JSON content
        json configData = json::parse(configFile);
        configFile.close();

        // Extract fields
        string server = configData["server"];
        int port = configData["port"];

        // Output in required format
        cout << "Configuration loaded: server=" << server << ", port=" << port << endl;

        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}