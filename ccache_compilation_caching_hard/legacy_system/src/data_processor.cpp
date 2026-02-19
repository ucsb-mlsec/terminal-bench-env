#include "data_processor.h"
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <memory>
#include <sstream>
#include <cmath>

namespace {
    template<typename T>
    struct is_numeric : std::integral_constant<bool,
        std::is_arithmetic<T>::value && !std::is_same<T, bool>::value> {};

    template<typename T>
    struct is_container {
        template<typename U>
        static auto test(int) -> decltype(
            std::declval<U>().begin(),
            std::declval<U>().end(),
            std::declval<U>().size(),
            std::true_type{});
        
        template<typename>
        static std::false_type test(...);
        
        static constexpr bool value = decltype(test<T>(0))::value;
    };
}

DataProcessor::DataProcessor() : initialized_(true), processing_count_(0) {
    config_["default_mode"] = "standard";
    config_["precision"] = "high";
}

DataProcessor::~DataProcessor() {
    cache_.clear();
    config_.clear();
}

template<typename T, typename Func>
std::vector<T> DataProcessor::transform(const std::vector<T>& input, Func&& func) {
    static_assert(std::is_invocable_v<Func, T>, "Function must be invocable with type T");
    std::vector<T> result;
    result.reserve(input.size());
    for (const auto& item : input) {
        result.push_back(std::forward<Func>(func)(item));
    }
    processing_count_++;
    return result;
}

template<typename T, typename Predicate>
std::vector<T> DataProcessor::filter(const std::vector<T>& input, Predicate&& pred) {
    static_assert(std::is_invocable_r_v<bool, Predicate, T>, 
                  "Predicate must return bool for type T");
    std::vector<T> result;
    std::copy_if(input.begin(), input.end(), std::back_inserter(result),
                 std::forward<Predicate>(pred));
    processing_count_++;
    return result;
}

template<typename T, typename BinaryOp>
typename std::enable_if<is_numeric<T>::value, T>::type
DataProcessor::aggregate(const std::vector<T>& input, T init, BinaryOp&& op) {
    processing_count_++;
    return std::accumulate(input.begin(), input.end(), init, std::forward<BinaryOp>(op));
}

template<typename T, typename U, typename Func>
std::vector<U> DataProcessor::map(const std::vector<T>& input, Func&& func) {
    static_assert(std::is_invocable_r_v<U, Func, T>, 
                  "Function must convert T to U");
    std::vector<U> result;
    result.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(result),
                   std::forward<Func>(func));
    processing_count_++;
    return result;
}

template<typename... Args>
auto DataProcessor::compose(Args&&... args) 
    -> decltype((args + ...)) {
    static_assert(sizeof...(Args) > 0, "Must provide at least one argument");
    return (args + ...);
}

template<typename T, typename... Funcs>
auto DataProcessor::applyPipeline(T value, Funcs&&... funcs) 
    -> decltype((funcs(value), ...)) {
    ((value = std::forward<Funcs>(funcs)(value)), ...);
    return value;
}

template<typename Container, typename Func>
typename std::enable_if<is_container<Container>::value, void>::type
DataProcessor::forEach(Container& container, Func&& func) {
    for (auto& item : container) {
        std::forward<Func>(func)(item);
    }
    processing_count_++;
}

template<typename T>
typename std::enable_if<is_numeric<T>::value, T>::type
DataProcessor::computeStatistic(const std::vector<T>& data, StatType type) {
    if (data.empty()) return T{};
    
    switch (type) {
        case StatType::MEAN: {
            T sum = std::accumulate(data.begin(), data.end(), T{});
            return sum / static_cast<T>(data.size());
        }
        case StatType::MEDIAN: {
            std::vector<T> sorted = data;
            std::sort(sorted.begin(), sorted.end());
            size_t mid = sorted.size() / 2;
            return sorted.size() % 2 == 0 
                ? (sorted[mid - 1] + sorted[mid]) / T{2}
                : sorted[mid];
        }
        case StatType::MAX:
            return *std::max_element(data.begin(), data.end());
        case StatType::MIN:
            return *std::min_element(data.begin(), data.end());
        default:
            return T{};
    }
}

template<typename Key, typename Value, typename Transformer>
std::map<Key, Value> DataProcessor::transformMap(
    const std::map<Key, Value>& input, Transformer&& trans) {
    std::map<Key, Value> result;
    for (const auto& [key, value] : input) {
        result[key] = std::forward<Transformer>(trans)(value);
    }
    processing_count_++;
    return result;
}

template<typename T, typename... Predicates>
bool DataProcessor::validateAll(const T& value, Predicates&&... preds) {
    return (std::forward<Predicates>(preds)(value) && ...);
}

template<typename T, typename... Predicates>
bool DataProcessor::validateAny(const T& value, Predicates&&... preds) {
    return (std::forward<Predicates>(preds)(value) || ...);
}

std::string DataProcessor::processData(const std::string& input) {
    std::stringstream ss;
    ss << "Processed: " << input << " [Count: " << processing_count_ << "]";
    processing_count_++;
    return ss.str();
}

bool DataProcessor::isInitialized() const {
    return initialized_;
}

size_t DataProcessor::getProcessingCount() const {
    return processing_count_;
}

void DataProcessor::reset() {
    processing_count_ = 0;
    cache_.clear();
}

template class std::vector<int>;
template class std::vector<double>;
template class std::vector<std::string>;