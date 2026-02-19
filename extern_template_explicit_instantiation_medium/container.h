#ifndef CONTAINER_H
#define CONTAINER_H

#include <vector>
#include <iostream>

template<typename T>
class Container {
private:
    std::vector<T> data;

public:
    Container() {}

    void add(const T& item) {
        data.push_back(item);
    }

    T get(size_t index) const {
        return data[index];
    }

    size_t size() const {
        return data.size();
    }

    void print() const {
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i];
            if (i < data.size() - 1) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
};

#endif