#ifndef VALIDATOR_H
#define VALIDATOR_H

#include <type_traits>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

class Validator {
public:
    Validator();
    ~Validator();

    // Template method with SFINAE for numeric types
    template<typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, bool>::type
    validate(const T& value, const T& min, const T& max) const {
        return value >= min && value <= max;
    }

    // Template method with SFINAE for container types
    template<typename Container>
    typename std::enable_if<
        std::is_class<Container>::value && 
        !std::is_same<Container, std::string>::value,
        bool
    >::type
    validate(const Container& container) const;

    // Constexpr validation for compile-time constants
    template<typename T>
    static constexpr bool check(T value, T threshold) {
        return value > threshold;
    }

    // Type trait based verification
    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    verify(const T& value) const {
        return value >= std::numeric_limits<T>::min() && 
               value <= std::numeric_limits<T>::max();
    }

    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    verify(const T& value) const {
        return !std::isnan(value) && !std::isinf(value);
    }

    // String validation
    bool validateString(const std::string& str, size_t minLength, size_t maxLength) const;

    // Range check with type traits
    template<typename T>
    static constexpr bool isInRange(T value, T lower, T upper) {
        static_assert(std::is_arithmetic<T>::value, "Type must be arithmetic");
        return value >= lower && value <= upper;
    }

    // Inline template implementation
    template<typename T>
    inline bool checkBounds(const std::vector<T>& vec, size_t index) const {
        return index < vec.size();
    }

    void setValidationMode(bool strict);
    bool getValidationMode() const;

private:
    bool strictMode;
    
    template<typename T>
    bool internalValidate(const T& value) const;
};

// Inline constexpr helper
template<typename T>
constexpr T clamp(T value, T min, T max) {
    return (value < min) ? min : (value > max) ? max : value;
}

#endif // VALIDATOR_H