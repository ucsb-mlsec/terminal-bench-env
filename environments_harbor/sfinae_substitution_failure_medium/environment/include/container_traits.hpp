#ifndef CONTAINER_TRAITS_HPP
#define CONTAINER_TRAITS_HPP

#include <type_traits>
#include <iterator>
#include <vector>

namespace container_utils {

// Primary template - types without iterator support
template<typename T, typename = void>
struct has_iterator : std::false_type {};

// Specialized template - types with begin() and end() methods
// BUG: Missing typename keyword and incorrect void_t usage
template<typename T>
struct has_iterator<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

// BUG: Missing typename keyword before enable_if::type
// Function to get container size using iterators
template<typename T>
std::enable_if<has_iterator<T>::value, size_t>::type
container_size(const T& c) {
    return std::distance(c.begin(), c.end());
}

// BUG: Incorrect enable_if syntax - should use ::type
// Function to check if container is empty
template<typename T>
std::enable_if<has_iterator<T>::value, bool>
is_empty(const T& c) {
    return c.begin() == c.end();
}

} // namespace container_utils

#endif // CONTAINER_TRAITS_HPP