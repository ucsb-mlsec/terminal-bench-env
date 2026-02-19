#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

// Computational model for statistical analysis and numerical simulations
// This model performs various mathematical operations on large datasets

// Template function for computing weighted averages
template<typename T>
T computeWeightedAverage(const std::vector<T>& values, const std::vector<T>& weights) {
    if (values.size() != weights.size() || values.empty()) {
        return T(0);
    }
    
    T weightedSum = T(0);
    T totalWeight = T(0);
    
    for (size_t i = 0; i < values.size(); ++i) {
        weightedSum += values[i] * weights[i];
        totalWeight += weights[i];
    }
    
    return (totalWeight != T(0)) ? weightedSum / totalWeight : T(0);
}

// Template function for matrix multiplication
template<typename T>
std::vector<std::vector<T>> matrixMultiply(
    const std::vector<std::vector<T>>& A,
    const std::vector<std::vector<T>>& B) {
    
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();
    
    std::vector<std::vector<T>> result(rowsA, std::vector<T>(colsB, T(0)));
    
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

// Statistical moments calculation
std::vector<double> computeStatisticalMoments(const std::vector<double>& data, int maxOrder) {
    std::vector<double> moments(maxOrder + 1, 0.0);
    
    if (data.empty()) {
        return moments;
    }
    
    // Compute mean (first moment)
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    moments[0] = mean;
    
    // Compute higher order central moments
    for (int order = 2; order <= maxOrder; ++order) {
        double sum = 0.0;
        for (const auto& value : data) {
            sum += std::pow(value - mean, order);
        }
        moments[order] = sum / data.size();
    }
    
    return moments;
}

// Numerical integration using trapezoidal rule
double trapezoidalIntegration(
    double (*func)(double),
    double a,
    double b,
    int n) {
    
    double h = (b - a) / n;
    double sum = 0.5 * (func(a) + func(b));
    
    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        sum += func(x);
    }
    
    return sum * h;
}

// Monte Carlo simulation for estimating pi
double monteCarloEstimatePi(int numSamples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    int insideCircle = 0;
    
    for (int i = 0; i < numSamples; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        
        if (x * x + y * y <= 1.0) {
            insideCircle++;
        }
    }
    
    return 4.0 * insideCircle / numSamples;
}

// Gradient descent optimization
std::vector<double> gradientDescent(
    std::vector<double> (*gradient)(const std::vector<double>&),
    std::vector<double> initialPoint,
    double learningRate,
    int maxIterations,
    double tolerance) {
    
    std::vector<double> point = initialPoint;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<double> grad = gradient(point);
        
        // Calculate gradient norm
        double gradNorm = 0.0;
        for (const auto& g : grad) {
            gradNorm += g * g;
        }
        gradNorm = std::sqrt(gradNorm);
        
        if (gradNorm < tolerance) {
            break;
        }
        
        // Update point
        for (size_t i = 0; i < point.size(); ++i) {
            point[i] -= learningRate * grad[i];
        }
    }
    
    return point;
}

// Compute correlation matrix
std::vector<std::vector<double>> computeCorrelationMatrix(
    const std::vector<std::vector<double>>& data) {
    
    size_t numVars = data.size();
    std::vector<std::vector<double>> correlation(
        numVars, std::vector<double>(numVars, 0.0));
    
    // Compute means
    std::vector<double> means(numVars);
    for (size_t i = 0; i < numVars; ++i) {
        means[i] = std::accumulate(data[i].begin(), data[i].end(), 0.0) / data[i].size();
    }
    
    // Compute standard deviations
    std::vector<double> stddevs(numVars);
    for (size_t i = 0; i < numVars; ++i) {
        double variance = 0.0;
        for (const auto& value : data[i]) {
            variance += std::pow(value - means[i], 2);
        }
        stddevs[i] = std::sqrt(variance / data[i].size());
    }
    
    // Compute correlations
    for (size_t i = 0; i < numVars; ++i) {
        for (size_t j = 0; j < numVars; ++j) {
            double covariance = 0.0;
            for (size_t k = 0; k < data[i].size(); ++k) {
                covariance += (data[i][k] - means[i]) * (data[j][k] - means[j]);
            }
            covariance /= data[i].size();
            correlation[i][j] = covariance / (stddevs[i] * stddevs[j]);
        }
    }
    
    return correlation;
}

// Example function for integration
double exampleFunction(double x) {
    return x * x + 2 * x + 1;
}

// Example gradient function for optimization
std::vector<double> exampleGradient(const std::vector<double>& point) {
    std::vector<double> grad(point.size());
    for (size_t i = 0; i < point.size(); ++i) {
        grad[i] = 2 * point[i]; // Gradient of x^2
    }
    return grad;
}

int main() {
    std::cout << "Starting computational model execution..." << std::endl;
    
    // Test weighted average
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> weights = {0.1, 0.2, 0.3, 0.2, 0.2};
    double weightedAvg = computeWeightedAverage(values, weights);
    std::cout << "Weighted average: " << weightedAvg << std::endl;
    
    // Test matrix multiplication
    std::vector<std::vector<double>> matA = {{1, 2}, {3, 4}};
    std::vector<std::vector<double>> matB = {{5, 6}, {7, 8}};
    auto matProduct = matrixMultiply(matA, matB);
    std::cout << "Matrix multiplication completed" << std::endl;
    
    // Test statistical moments
    std::vector<double> dataset = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto moments = computeStatisticalMoments(dataset, 4);
    std::cout << "Mean: " << moments[0] << std::endl;
    
    // Test numerical integration
    double integral = trapezoidalIntegration(exampleFunction, 0.0, 1.0, 1000);
    std::cout << "Integral result: " << integral << std::endl;
    
    // Test Monte Carlo simulation
    double piEstimate = monteCarloEstimatePi(100000);
    std::cout << "Pi estimate: " << piEstimate << std::endl;
    
    // Test gradient descent
    std::vector<double> initialPoint = {10.0, 10.0};
    auto optimized = gradientDescent(exampleGradient, initialPoint, 0.1, 100, 1e-6);
    std::cout << "Optimization completed" << std::endl;
    
    // Test correlation matrix
    std::vector<std::vector<double>> multivarData = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {2.0, 4.0, 6.0, 8.0, 10.0},
        {1.5, 3.0, 4.5, 6.0, 7.5}
    };
    auto corrMatrix = computeCorrelationMatrix(multivarData);
    std::cout << "Correlation matrix computed" << std::endl;
    
    std::cout << "All computational tests completed successfully!" << std::endl;
    
    return 0;
}