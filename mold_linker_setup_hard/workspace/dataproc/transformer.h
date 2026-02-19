#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace dataproc {

// String transformation functions
std::string trim(const std::string& str);
std::string toUpperCase(const std::string& str);
std::string toLowerCase(const std::string& str);
std::string sanitize(const std::string& str);

// Type conversion functions
int stringToInt(const std::string& str);
double stringToDouble(const std::string& str);
bool stringToBool(const std::string& str);
std::string intToString(int value);
std::string doubleToString(double value);

// Derived field computation functions
double calculatePercentage(double value, double total);
double calculateAverage(const std::vector<double>& values);
double calculateSum(const std::vector<double>& values);
std::string concatenateFields(const std::vector<std::string>& fields, const std::string& delimiter);

// Transformation pipeline types
using TransformFunc = std::function<std::string(const std::string&)>;
using FieldTransformer = std::function<std::vector<std::string>(const std::vector<std::string>&)>;

class TransformationPipeline {
public:
    TransformationPipeline();
    void addTransform(TransformFunc func);
    std::string apply(const std::string& input) const;
    void clear();
    
private:
    std::vector<TransformFunc> transforms;
};

} // namespace dataproc

#endif // TRANSFORMER_H