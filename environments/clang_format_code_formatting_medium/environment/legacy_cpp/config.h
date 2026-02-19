#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <map>

namespace legacy {

class Configuration {
public:
    Configuration();
    ~Configuration();

    bool load(const std::string& filename);
    bool save(const std::string& filename);

    void set(const std::string& key, const std::string& value);
    std::string get(const std::string& key) const;
    bool has(const std::string& key) const;

    void setInt(const std::string& key, int value);
    int getInt(const std::string& key, int defaultValue = 0) const;

    void setBool(const std::string& key, bool value);
    bool getBool(const std::string& key, bool defaultValue = false) const;

    void clear();
    size_t size() const;

private:
    std::map<std::string, std::string> m_settings;
    std::string m_filename;

    std::string trim(const std::string& str) const;
    bool parseLine(const std::string& line);
};

bool initializeDefaultConfig(Configuration& config);
bool validateConfiguration(const Configuration& config);
std::string getConfigPath();

}

#endif