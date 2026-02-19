#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <cmath>
#include <algorithm>
#include <random>

class OptimizationEngine {
private:
    std::vector<double> parameters;
    double learning_rate;
    int max_iterations;

public:
    OptimizationEngine(size_t dim, double lr = 0.01, int max_iter = 1000) 
        : parameters(dim, 0.0), learning_rate(lr), max_iterations(max_iter) {}

    double objective_function(const std::vector<double>& x) {
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i] - 10.0 * std::cos(2.0 * M_PI * x[i]);
        }
        return 10.0 * x.size() + sum;
    }

    std::vector<double> gradient(const std::vector<double>& x) {
        std::vector<double> grad(x.size());
        const double h = 1e-5;
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> x_plus = x;
            std::vector<double> x_minus = x;
            x_plus[i] += h;
            x_minus[i] -= h;
            grad[i] = (objective_function(x_plus) - objective_function(x_minus)) / (2.0 * h);
        }
        return grad;
    }

    void optimize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-5.0, 5.0);
        
        for (auto& p : parameters) {
            p = dis(gen);
        }

        for (int iter = 0; iter < max_iterations; ++iter) {
            std::vector<double> grad = gradient(parameters);
            for (size_t i = 0; i < parameters.size(); ++i) {
                parameters[i] -= learning_rate * grad[i];
            }
            
            if (iter % 100 == 0) {
                double cost = objective_function(parameters);
                if (cost < 0.01) break;
            }
        }
    }

    const std::vector<double>& get_parameters() const {
        return parameters;
    }
};

class SignalProcessor {
private:
    std::vector<double> signal;
    size_t window_size;

public:
    SignalProcessor(const std::vector<double>& input_signal, size_t ws = 5) 
        : signal(input_signal), window_size(ws) {}

    std::vector<double> moving_average_filter() {
        std::vector<double> filtered(signal.size());
        for (size_t i = 0; i < signal.size(); ++i) {
            double sum = 0.0;
            int count = 0;
            for (size_t j = 0; j < window_size; ++j) {
                int idx = i - window_size/2 + j;
                if (idx >= 0 && idx < static_cast<int>(signal.size())) {
                    sum += signal[idx];
                    count++;
                }
            }
            filtered[i] = count > 0 ? sum / count : 0.0;
        }
        return filtered;
    }

    std::vector<double> compute_spectrum() {
        std::vector<double> spectrum(signal.size() / 2);
        for (size_t k = 0; k < spectrum.size(); ++k) {
            double real_part = 0.0;
            double imag_part = 0.0;
            for (size_t n = 0; n < signal.size(); ++n) {
                double angle = -2.0 * M_PI * k * n / signal.size();
                real_part += signal[n] * std::cos(angle);
                imag_part += signal[n] * std::sin(angle);
            }
            spectrum[k] = std::sqrt(real_part * real_part + imag_part * imag_part);
        }
        return spectrum;
    }
};

std::vector<double> generate_synthetic_data(size_t n, double noise_level = 0.1) {
    std::vector<double> data(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_dist(0.0, noise_level);
    
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / n;
        data[i] = std::sin(2.0 * M_PI * 5.0 * t) + 
                  0.5 * std::cos(2.0 * M_PI * 10.0 * t) + 
                  noise_dist(gen);
    }
    return data;
}

void perform_statistical_analysis(const std::vector<double>& data) {
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    
    double variance = 0.0;
    for (const auto& val : data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= data.size();
    double std_dev = std::sqrt(variance);
    
    auto minmax = std::minmax_element(data.begin(), data.end());
    double min_val = *minmax.first;
    double max_val = *minmax.second;
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    double median = sorted_data[sorted_data.size() / 2];
    
    std::cout << "Statistical Analysis Results:" << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std Dev: " << std_dev << std::endl;
    std::cout << "  Median: " << median << std::endl;
    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
}

void run_data_analysis_pipeline() {
    std::cout << "Running Data Analysis Pipeline..." << std::endl;
    
    const size_t data_size = 500;
    std::vector<double> raw_data = generate_synthetic_data(data_size, 0.15);
    
    SignalProcessor processor(raw_data, 7);
    std::vector<double> filtered = processor.moving_average_filter();
    std::vector<double> spectrum = processor.compute_spectrum();
    
    perform_statistical_analysis(filtered);
    
    double spectral_energy = std::accumulate(spectrum.begin(), spectrum.end(), 0.0,
        [](double acc, double val) { return acc + val * val; });
    
    std::cout << "  Spectral Energy: " << spectral_energy << std::endl;
}

int main() {
    std::cout << "Computational Model 2: Optimization and Signal Processing" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    OptimizationEngine optimizer(8, 0.001, 500);
    std::cout << "\nRunning optimization engine..." << std::endl;
    optimizer.optimize();
    
    const auto& optimal_params = optimizer.get_parameters();
    std::cout << "Optimization complete. Final parameters: ";
    for (size_t i = 0; i < optimal_params.size(); ++i) {
        std::cout << optimal_params[i];
        if (i < optimal_params.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    run_data_analysis_pipeline();
    
    std::cout << "\nModel 2 execution completed successfully." << std::endl;
    
    return 0;
}