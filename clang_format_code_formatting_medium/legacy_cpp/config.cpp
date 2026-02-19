#include <iostream>
#include <string>
#include <map>
#include <fstream>

class Configuration {
private:
    std::map<std::string, std::string> settings;
    std::string configFile;

public:
    Configuration(const std::string& filename) : configFile(filename) {
        loadConfiguration();
    }

    bool loadConfiguration() {
        std::ifstream file(configFile);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                settings[key] = value;
            }
        }
        file.close();
        return true;
    }

    std::string getValue(const std::string& key) const {
        auto it = settings.find(key);
        if (it != settings.end()) {
            return it->second;
        }
        return "";
    }

    void setValue(const std::string& key, const std::string& value) {
        settings[key] = value;
    }

    bool saveConfiguration() const {
        std::ofstream file(configFile);
        if (!file.is_open()) {
            return false;
        }

        for (const auto& pair : settings) {
            file << pair.first << "=" << pair.second << std::endl;
        }
        file.close();
        return true;
    }

    void displayAll() const {
        std::cout << "Current Configuration:" << std::endl;
        for (const auto& pair : settings) {
            std::cout << "  " << pair.first << " = " << pair.second << std::endl;
        }
    }
};