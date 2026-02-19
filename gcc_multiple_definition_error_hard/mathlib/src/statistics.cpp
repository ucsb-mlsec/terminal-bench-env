#include "statistics.h"
#include "utils.h"
#include "constants.h"
#include <cmath>

namespace MathLib {

double Stats::mean(const Vector& v) {
    statsCallCount++;
    if (v.size() == 0) {
        return 0.0;
    }
    return sumElements(v) / v.size();
}

double Stats::variance(const Vector& v) {
    if (v.size() == 0) {
        return 0.0;
    }
    double m = mean(v);
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        double diff = v[i] - m;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / v.size();
}

double Stats::stddev(const Vector& v) {
    return std::sqrt(variance(v));
}

} // namespace MathLib