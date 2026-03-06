#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H

#include <vector>
#include <string>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <memory>

namespace legacy {

// Forward declarations
template<typename T>
class ProcessorConfig;

// SFINAE helper for numeric types
template<typename T>
using enable_if_numeric = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;

// SFINAE helper for container types
template<typename T>
struct is_container {
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().begin(),
        std::declval<U>().end(),
        std::true_type{}
    );
    
    template<typename>
    static std::false_type test(...);
    
    static constexpr bool value = decltype(test<T>(0))::value;
};

// Main DataProcessor class with template specializations
template<typename T, typename Enable = void>
class DataProcessor {
public:
    DataProcessor();
    explicit DataProcessor(const std::vector<T>& data);
    ~DataProcessor();
    
    // Template methods for transformation
    template<typename Func>
    std::vector<T> transform(Func&& func) const;
    
    template<typename Pred>
    std::vector<T> filter(Pred&& predicate) const;
    
    template<typename U, typename Reducer>
    U aggregate(U initial, Reducer&& reducer) const;
    
    // Variadic template for multi-step processing
    template<typename... Funcs>
    std::vector<T> pipeline(Funcs&&... functions) const;
    
    void setData(const std::vector<T>& data);
    const std::vector<T>& getData() const;
    size_t size() const;
    
private:
    std::vector<T> m_data;
    
    template<typename Func>
    void applyFunction(Func&& func);
};

// Specialization for numeric types
template<typename T>
class DataProcessor<T, enable_if_numeric<T>> {
public:
    DataProcessor() : m_data(), m_sum(0) {}
    explicit DataProcessor(const std::vector<T>& data) : m_data(data), m_sum(0) {
        computeSum();
    }
    
    T sum() const { return m_sum; }
    T mean() const { return m_data.empty() ? T{} : m_sum / m_data.size(); }
    
    template<typename Func>
    inline std::vector<T> transform(Func&& func) const {
        std::vector<T> result;
        result.reserve(m_data.size());
        for (const auto& item : m_data) {
            result.push_back(func(item));
        }
        return result;
    }
    
    template<typename Pred>
    inline std::vector<T> filter(Pred&& predicate) const {
        std::vector<T> result;
        std::copy_if(m_data.begin(), m_data.end(), std::back_inserter(result), predicate);
        return result;
    }
    
    void setData(const std::vector<T>& data) {
        m_data = data;
        computeSum();
    }
    
private:
    std::vector<T> m_data;
    T m_sum;
    
    void computeSum() {
        m_sum = std::accumulate(m_data.begin(), m_data.end(), T{});
    }
};

// Inline template implementations for general case
template<typename T, typename Enable>
template<typename Func>
inline std::vector<T> DataProcessor<T, Enable>::transform(Func&& func) const {
    std::vector<T> result;
    result.reserve(m_data.size());
    std::transform(m_data.begin(), m_data.end(), std::back_inserter(result), std::forward<Func>(func));
    return result;
}

template<typename T, typename Enable>
template<typename Pred>
inline std::vector<T> DataProcessor<T, Enable>::filter(Pred&& predicate) const {
    std::vector<T> result;
    std::copy_if(m_data.begin(), m_data.end(), std::back_inserter(result), std::forward<Pred>(predicate));
    return result;
}

} // namespace legacy

#endif // DATA_PROCESSOR_H