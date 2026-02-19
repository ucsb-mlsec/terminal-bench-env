#ifndef FILTER_H
#define FILTER_H

#include <vector>
#include <string>
#include <functional>

namespace dataproc {

// Filter types
enum class FilterType {
    NUMERIC_RANGE,
    STRING_MATCH,
    REGEX_PATTERN,
    CUSTOM_PREDICATE
};

// Filter result structure
struct FilterResult {
    size_t matched_count;
    size_t total_count;
    double match_percentage;
};

// Type definitions for filter predicates
using NumericPredicate = std::function<bool(double)>;
using StringPredicate = std::function<bool(const std::string&)>;
using RowPredicate = std::function<bool(const std::vector<std::string>&)>;

// Filter function declarations
FilterResult filterNumericColumn(const std::vector<double>& data, 
                                 NumericPredicate predicate);

FilterResult filterStringColumn(const std::vector<std::string>& data,
                                const std::string& pattern,
                                bool case_sensitive = true);

FilterResult filterRows(const std::vector<std::vector<std::string>>& rows,
                        RowPredicate predicate);

bool applyRangeFilter(double value, double min, double max);

std::vector<size_t> getMatchingIndices(const std::vector<std::string>& data,
                                       const std::string& search_term);

} // namespace dataproc

#endif // FILTER_H