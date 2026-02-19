#ifndef COMPUTE_ENGINE_H
#define COMPUTE_ENGINE_H

#include <vector>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace legacy {

template<typename T, typename U = double, std::size_t BufferSize = 1024>
class ComputeEngine {
public:
    using value_type = T;
    using precision_type = U;
    using container_type = std::vector<T>;
    
    ComputeEngine();
    explicit ComputeEngine(std::size_t capacity);
    ~ComputeEngine() = default;
    
    template<typename InputIterator>
    U calculate(InputIterator first, InputIterator last);
    
    template<typename Predicate>
    container_type process(const container_type& input, Predicate pred);
    
    template<typename Transform>
    U optimize(const container_type& data, Transform transform);
    
    void initialize(std::size_t size);
    void reset();
    std::size_t size() const noexcept { return m_data.size(); }
    bool empty() const noexcept { return m_data.empty(); }
    
    template<typename Func>
    auto apply(Func f) -> typename std::result_of<Func(T)>::type;
    
    template<typename... Args>
    void configure(Args&&... args);

private:
    container_type m_data;
    std::unique_ptr<U[]> m_buffer;
    std::size_t m_capacity;
    
    template<typename V>
    typename std::enable_if<std::is_arithmetic<V>::value, V>::type
    compute_internal(V value);
};

template<typename T, typename U, std::size_t BufferSize>
ComputeEngine<T, U, BufferSize>::ComputeEngine() 
    : m_capacity(BufferSize), m_buffer(new U[BufferSize]) {
    m_data.reserve(BufferSize);
}

template<typename T, typename U, std::size_t BufferSize>
ComputeEngine<T, U, BufferSize>::ComputeEngine(std::size_t capacity)
    : m_capacity(capacity), m_buffer(new U[capacity]) {
    m_data.reserve(capacity);
}

template<typename T, typename U, std::size_t BufferSize>
template<typename InputIterator>
U ComputeEngine<T, U, BufferSize>::calculate(InputIterator first, InputIterator last) {
    U sum = U();
    std::size_t count = 0;
    for (auto it = first; it != last; ++it, ++count) {
        sum += static_cast<U>(*it);
    }
    return count > 0 ? sum / static_cast<U>(count) : U();
}

template<typename T, typename U, std::size_t BufferSize>
template<typename Predicate>
typename ComputeEngine<T, U, BufferSize>::container_type 
ComputeEngine<T, U, BufferSize>::process(const container_type& input, Predicate pred) {
    container_type result;
    result.reserve(input.size());
    std::copy_if(input.begin(), input.end(), std::back_inserter(result), pred);
    return result;
}

template<typename T, typename U, std::size_t BufferSize>
template<typename Transform>
U ComputeEngine<T, U, BufferSize>::optimize(const container_type& data, Transform transform) {
    if (data.empty()) return U();
    U result = transform(data[0]);
    for (std::size_t i = 1; i < data.size(); ++i) {
        result = std::max(result, transform(data[i]));
    }
    return result;
}

template<typename T, typename U, std::size_t BufferSize>
void ComputeEngine<T, U, BufferSize>::initialize(std::size_t size) {
    m_data.clear();
    m_data.resize(size);
}

template<typename T, typename U, std::size_t BufferSize>
void ComputeEngine<T, U, BufferSize>::reset() {
    m_data.clear();
}

} // namespace legacy

#endif // COMPUTE_ENGINE_H