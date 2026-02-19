#include "io_handler.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <type_traits>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

// Type trait to check if a type is serializable
template<typename T>
struct is_trivially_serializable {
    static constexpr bool value = std::is_arithmetic<T>::value || std::is_enum<T>::value;
};

// Compile-time size calculation for serialization buffer
template<typename T>
constexpr size_t serialization_size() {
    return sizeof(T);
}

template<typename T>
constexpr size_t serialization_size(const T& value) {
    return sizeof(T);
}

IOHandler::IOHandler() : buffer_size_(0), verbose_(false) {
    buffer_.reserve(DEFAULT_BUFFER_SIZE);
}

IOHandler::~IOHandler() {
    if (output_stream_.is_open()) {
        output_stream_.close();
    }
    if (input_stream_.is_open()) {
        input_stream_.close();
    }
}

bool IOHandler::openFile(const std::string& filename, bool write_mode) {
    if (write_mode) {
        output_stream_.open(filename, std::ios::binary);
        return output_stream_.is_open();
    } else {
        input_stream_.open(filename, std::ios::binary);
        return input_stream_.is_open();
    }
}

void IOHandler::closeFile() {
    if (output_stream_.is_open()) {
        output_stream_.close();
    }
    if (input_stream_.is_open()) {
        input_stream_.close();
    }
}

// Template method for reading primitive types
template<typename T>
typename std::enable_if<is_trivially_serializable<T>::value, bool>::type
IOHandler::read(T& value) {
    if (!input_stream_.is_open() || !input_stream_.good()) {
        return false;
    }
    input_stream_.read(reinterpret_cast<char*>(&value), sizeof(T));
    return input_stream_.good();
}

// Template method for writing primitive types
template<typename T>
typename std::enable_if<is_trivially_serializable<T>::value, bool>::type
IOHandler::write(const T& value) {
    if (!output_stream_.is_open()) {
        return false;
    }
    output_stream_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return output_stream_.good();
}

// Template serialization for arithmetic types
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::string>::type
IOHandler::serialize(const T& value) {
    std::ostringstream oss;
    oss.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return oss.str();
}

// Template deserialization for arithmetic types
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
IOHandler::deserialize(const std::string& data) {
    if (data.size() < sizeof(T)) {
        throw std::runtime_error("Insufficient data for deserialization");
    }
    T value;
    std::memcpy(&value, data.data(), sizeof(T));
    return value;
}

// Specialized template for vector serialization
template<typename T>
std::string IOHandler::serializeVector(const std::vector<T>& vec) {
    static_assert(is_trivially_serializable<T>::value, 
                  "Type must be trivially serializable");
    
    std::ostringstream oss;
    size_t size = vec.size();
    oss.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    for (const auto& item : vec) {
        oss.write(reinterpret_cast<const char*>(&item), sizeof(T));
    }
    return oss.str();
}

// Specialized template for vector deserialization
template<typename T>
std::vector<T> IOHandler::deserializeVector(const std::string& data) {
    static_assert(is_trivially_serializable<T>::value,
                  "Type must be trivially serializable");
    
    std::istringstream iss(data);
    size_t size;
    iss.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    std::vector<T> vec(size);
    for (size_t i = 0; i < size; ++i) {
        iss.read(reinterpret_cast<char*>(&vec[i]), sizeof(T));
    }
    return vec;
}

// Explicit template instantiations
template bool IOHandler::read<int>(int&);
template bool IOHandler::read<double>(double&);
template bool IOHandler::read<float>(float&);
template bool IOHandler::read<long>(long&);

template bool IOHandler::write<int>(const int&);
template bool IOHandler::write<double>(const double&);
template bool IOHandler::write<float>(const float&);
template bool IOHandler::write<long>(const long&);

template std::string IOHandler::serialize<int>(const int&);
template std::string IOHandler::serialize<double>(const double&);
template std::string IOHandler::serialize<float>(const float&);

template int IOHandler::deserialize<int>(const std::string&);
template double IOHandler::deserialize<double>(const std::string&);
template float IOHandler::deserialize<float>(const std::string&);