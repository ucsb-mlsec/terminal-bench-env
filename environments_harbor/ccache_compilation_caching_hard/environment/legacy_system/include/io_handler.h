#ifndef IO_HANDLER_H
#define IO_HANDLER_H

#include <iostream>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>
#include <memory>
#include <sstream>

namespace legacy {

class IOHandler {
public:
    IOHandler();
    ~IOHandler();

    // Template method for reading with SFINAE
    template<typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    read(std::istream& stream);

    template<typename T>
    typename std::enable_if<!std::is_arithmetic<T>::value, T>::type
    read(std::istream& stream);

    // Template method for writing with SFINAE
    template<typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, void>::type
    write(std::ostream& stream, const T& value);

    template<typename T>
    typename std::enable_if<!std::is_arithmetic<T>::value, void>::type
    write(std::ostream& stream, const T& value);

    // Serialization with complex template parameters
    template<typename T, typename Allocator = std::allocator<T>>
    std::string serialize(const std::vector<T, Allocator>& data);

    template<typename T>
    std::string serialize(const T& data);

    // Deserialization with type traits
    template<typename T>
    typename std::enable_if<std::is_pod<T>::value, T>::type
    deserialize(const std::string& data);

    template<typename T>
    typename std::enable_if<!std::is_pod<T>::value, T>::type
    deserialize(const std::string& data);

    // Inline template implementations
    template<typename Container>
    inline size_t getSize(const Container& c) {
        return c.size();
    }

    template<typename T>
    inline bool validate(const T& value) {
        return sizeof(T) > 0;
    }

private:
    std::string buffer_;
    size_t position_;
};

// Inline template method implementations
template<typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
IOHandler::read(std::istream& stream) {
    T value;
    stream >> value;
    return value;
}

template<typename T>
inline std::string IOHandler::serialize(const T& data) {
    std::ostringstream oss;
    oss << data;
    return oss.str();
}

template<typename T, typename Allocator>
std::string IOHandler::serialize(const std::vector<T, Allocator>& data) {
    std::ostringstream oss;
    for (const auto& item : data) {
        oss << item << " ";
    }
    return oss.str();
}

} // namespace legacy

#endif // IO_HANDLER_H