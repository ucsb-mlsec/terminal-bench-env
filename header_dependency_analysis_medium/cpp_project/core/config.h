#ifndef CONFIG_H
#define CONFIG_H

#include "settings.h"
#include <string>
#include <map>

namespace core {

class Config {
public:
    Config();
    ~Config();
    
    bool load(const std::string& filename);
    bool save(const std::string& filename);
    
    std::string get(const std::string& key) const;
    void set(const std::string& key, const std::string& value);
    
    bool has(const std::string& key) const;
    void clear();
    
private:
    std::map<std::string, std::string> data_;
    std::string filename_;
    bool modified_;
};

} // namespace core

#endif // CONFIG_H