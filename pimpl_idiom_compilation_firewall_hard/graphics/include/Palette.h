#ifndef PALETTE_H
#define PALETTE_H

#include <map>
#include <string>
#include <vector>
#include <memory>

class Palette {
public:
    Palette();
    ~Palette();
    void addColor(const std::string& name, unsigned int color);
    unsigned int getColor(const std::string& name) const;
    int getColorCount() const;

private:
    std::map<std::string, unsigned int> m_colors;
    std::vector<std::string> m_colorNames;
    std::unique_ptr<int> m_metadata;
};

#endif