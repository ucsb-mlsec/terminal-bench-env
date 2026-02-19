#include <boost/python.hpp>
#include <vector>

using namespace boost::python;

// Simple matrix operations for demonstration
// Matrices represented as flat vectors with dimensions

// Multiply each element of a matrix by a scalar
double multiply_scalar(double value, double scalar) {
    return value * scalar;
}

// Add two numbers (simplified matrix element operation)
double add_elements(double a, double b) {
    return a + b;
}

// Matrix transpose simulation - swap dimensions
int get_transposed_index(int row, int col, int num_cols) {
    return col * num_cols + row;
}

// Calculate matrix determinant for 2x2 matrix
double determinant_2x2(double a, double b, double c, double d) {
    return a * d - b * c;
}

// Dot product of two vectors
double dot_product(list py_list1, list py_list2) {
    double result = 0.0;
    int len1 = len(py_list1);
    int len2 = len(py_list2);
    
    if (len1 != len2) {
        return 0.0;
    }
    
    for (int i = 0; i < len1; i++) {
        result += extract<double>(py_list1[i]) * extract<double>(py_list2[i]);
    }
    
    return result;
}

// Module definition with INCORRECT name - this causes the segfault!
// The setup.py expects 'matrix_ops' but this says 'matrix_extension'
BOOST_PYTHON_MODULE(matrix_extension) {
    // Export our matrix operation functions
    def("multiply_scalar", multiply_scalar,
        "Multiply a value by a scalar");
    
    def("add_elements", add_elements,
        "Add two matrix elements");
    
    def("get_transposed_index", get_transposed_index,
        "Calculate transposed matrix index");
    
    def("determinant_2x2", determinant_2x2,
        "Calculate 2x2 matrix determinant");
    
    def("dot_product", dot_product,
        "Calculate dot product of two vectors");
}