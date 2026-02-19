#include "Palette.h"
#include <map>
#include <vector>
#include <string>
#include <memory>

struct Palette::Impl {
    std::map<std::string, unsigned int> m_colors;
    std::vector<std::string> m_colorNames;
    std::unique_ptr<int> m_metadata;
    
    Impl() : m_metadata(new int(0)) {}
};

Palette::Palette() : pImpl(new Impl()) {
}

Palette::~Palette() {
    delete pImpl;
}

void Palette::addColor(const std::string& name, unsigned int color) {
    pImpl->m_colors[name] = color;
    pImpl->m_colorNames.push_back(name);
}

unsigned int Palette::getColor(const std::string& name) const {
    auto it = pImpl->m_colors.find(name);
    if (it != pImpl->m_colors.end()) {
        return it->second;
    }
    return 0;
}

int Palette::getColorCount() const {
    return pImpl->m_colors.size();
}