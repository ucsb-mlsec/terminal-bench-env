#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "vector_ops.hpp"

bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    std::cout << "Testing numlib vector operations..." << std::endl;
    
    // Test 1: Add double vectors
    std::cout << "\nTest 1: Adding double vectors" << std::endl;
    std::vector<double> v1_d = {1.0, 2.0, 3.0};
    std::vector<double> v2_d = {4.0, 5.0, 6.0};
    std::vector<double> result_add_d = add(v1_d, v2_d);
    
    assert(result_add_d.size() == 3);
    assert(approx_equal(result_add_d[0], 5.0));
    assert(approx_equal(result_add_d[1], 7.0));
    assert(approx_equal(result_add_d[2], 9.0));
    std::cout << "  {1.0, 2.0, 3.0} + {4.0, 5.0, 6.0} = {5.0, 7.0, 9.0} ✓" << std::endl;
    
    // Test 2: Add integer vectors
    std::cout << "\nTest 2: Adding integer vectors" << std::endl;
    std::vector<int> v1_i = {10, 20, 30, 40};
    std::vector<int> v2_i = {5, 15, 25, 35};
    std::vector<int> result_add_i = add(v1_i, v2_i);
    
    assert(result_add_i.size() == 4);
    assert(result_add_i[0] == 15);
    assert(result_add_i[1] == 35);
    assert(result_add_i[2] == 55);
    assert(result_add_i[3] == 75);
    std::cout << "  {10, 20, 30, 40} + {5, 15, 25, 35} = {15, 35, 55, 75} ✓" << std::endl;
    
    // Test 3: Multiply integer vector by scalar
    std::cout << "\nTest 3: Multiplying integer vector by scalar" << std::endl;
    std::vector<int> v_mult_i = {2, 4, 6};
    std::vector<int> result_mult_i = multiply(v_mult_i, 3);
    
    assert(result_mult_i.size() == 3);
    assert(result_mult_i[0] == 6);
    assert(result_mult_i[1] == 12);
    assert(result_mult_i[2] == 18);
    std::cout << "  {2, 4, 6} * 3 = {6, 12, 18} ✓" << std::endl;
    
    // Test 4: Multiply double vector by scalar
    std::cout << "\nTest 4: Multiplying double vector by scalar" << std::endl;
    std::vector<double> v_mult_d = {1.5, 2.5, 3.5, 4.5};
    std::vector<double> result_mult_d = multiply(v_mult_d, 2.0);
    
    assert(result_mult_d.size() == 4);
    assert(approx_equal(result_mult_d[0], 3.0));
    assert(approx_equal(result_mult_d[1], 5.0));
    assert(approx_equal(result_mult_d[2], 7.0));
    assert(approx_equal(result_mult_d[3], 9.0));
    std::cout << "  {1.5, 2.5, 3.5, 4.5} * 2.0 = {3.0, 5.0, 7.0, 9.0} ✓" << std::endl;
    
    // Test 5: Dot product of double vectors
    std::cout << "\nTest 5: Dot product of double vectors" << std::endl;
    std::vector<double> v_dot1 = {1.0, 2.0, 3.0};
    std::vector<double> v_dot2 = {4.0, 5.0, 6.0};
    double dot_result = dot_product(v_dot1, v_dot2);
    
    assert(approx_equal(dot_result, 32.0));  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    std::cout << "  {1.0, 2.0, 3.0} · {4.0, 5.0, 6.0} = 32.0 ✓" << std::endl;
    
    // Test 6: Dot product of integer vectors
    std::cout << "\nTest 6: Dot product of integer vectors" << std::endl;
    std::vector<int> v_dot_i1 = {1, 2, 3, 4};
    std::vector<int> v_dot_i2 = {2, 3, 4, 5};
    int dot_result_i = dot_product(v_dot_i1, v_dot_i2);
    
    assert(dot_result_i == 40);  // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    std::cout << "  {1, 2, 3, 4} · {2, 3, 4, 5} = 40 ✓" << std::endl;
    
    std::cout << "\n✓ All tests passed successfully!" << std::endl;
    
    return 0;
}