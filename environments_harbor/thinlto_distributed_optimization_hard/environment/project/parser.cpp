#include <iostream>
#include <fstream>
#include <string>
#include "parser.h"

bool parseConfig(const std::string& configPath, ConfigData& config) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << configPath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t delimPos = line.find('=');
        if (delimPos == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, delimPos);
        std::string value = line.substr(delimPos + 1);

        if (key == "threads") {
            config.threadCount = std::stoi(value);
        } else if (key == "buffer_size") {
            config.bufferSize = std::stoi(value);
        } else if (key == "output_path") {
            config.outputPath = value;
        }
    }

    file.close();
    return true;
}

bool parseInputFile(const std::string& inputPath, DataSet& dataset) {
    std::ifstream file(inputPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open input file: " << inputPath << std::endl;
        return false;
    }

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line)) {
        lineNumber++;
        if (line.empty()) {
            continue;
        }

        DataEntry entry;
        size_t pos = 0;
        size_t nextPos = line.find(',');

        if (nextPos != std::string::npos) {
            entry.id = std::stoi(line.substr(pos, nextPos - pos));
            pos = nextPos + 1;
            nextPos = line.find(',', pos);

            if (nextPos != std::string::npos) {
                entry.name = line.substr(pos, nextPos - pos);
                pos = nextPos + 1;
                entry.value = std::stod(line.substr(pos));

                dataset.entries.push_back(entry);
            }
        }
    }

    file.close();
    dataset.count = dataset.entries.size();
    return dataset.count > 0;
}

bool validateData(const DataSet& dataset, const ConfigData& config) {
    if (dataset.entries.empty()) {
        std::cerr << "Dataset is empty" << std::endl;
        return false;
    }

    if (config.threadCount <= 0 || config.threadCount > 128) {
        std::cerr << "Invalid thread count: " << config.threadCount << std::endl;
        return false;
    }

    if (config.bufferSize <= 0) {
        std::cerr << "Invalid buffer size: " << config.bufferSize << std::endl;
        return false;
    }

    for (const auto& entry : dataset.entries) {
        if (entry.id < 0) {
            std::cerr << "Invalid entry ID: " << entry.id << std::endl;
            return false;
        }
        if (entry.name.empty()) {
            std::cerr << "Empty name in entry ID: " << entry.id << std::endl;
            return false;
        }
    }

    return true;
}