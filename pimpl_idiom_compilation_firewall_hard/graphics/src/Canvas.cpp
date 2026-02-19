#include "Canvas.h"
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

struct Canvas::Impl {
    int m_width;
    int m_height;
    std::vector<unsigned int> m_pixels;
    std::string m_name;
    std::unique_ptr<int> m_metadata;
    
    Impl(int width, int height)
        : m_width(width)
        , m_height(height)
        , m_pixels(width * height, 0)
        , m_name("canvas")
        , m_metadata(new int(0))
    {
    }
};

Canvas::Canvas(int width, int height)
    : pImpl(new Impl(width, height))
{
}

Canvas::~Canvas() {
    delete pImpl;
}

void Canvas::clear() {
    std::fill(pImpl->m_pixels.begin(), pImpl->m_pixels.end(), 0);
}

void Canvas::setPixel(int x, int y, unsigned int color) {
    if (x >= 0 && x < pImpl->m_width && y >= 0 && y < pImpl->m_height) {
        pImpl->m_pixels[y * pImpl->m_width + x] = color;
    }
}

int Canvas::getWidth() const {
    return pImpl->m_width;
}

int Canvas::getHeight() const {
    return pImpl->m_height;
}