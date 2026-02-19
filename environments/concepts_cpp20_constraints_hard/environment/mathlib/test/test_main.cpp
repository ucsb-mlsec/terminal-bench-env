#include "../include/operations.hpp"
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <cassert>

int main() {
    std::cout << "Starting mathlib tests...\n\n";

    // Test 1: Basic integral type addition
    std::cout << "Test 1: sum(5, 10) = ";
    auto result1 = sum(5, 10);
    std::cout << result1 << " (expected: 15)\n";
    assert(result1 == 15);

    // Test 2: Long integer multiplication
    std::cout << "Test 2: multiply(3L, 7L) = ";
    auto result2 = multiply(3L, 7L);
    std::cout << result2 << " (expected: 21)\n";
    assert(result2 == 21);

    // Test 3: Floating point addition - will fail due to restrictive Numeric concept
    std::cout << "Test 3: sum(3.14, 2.86) = ";
    auto result3 = sum(3.14, 2.86);
    std::cout << result3 << " (expected: ~6.0)\n";
    assert(result3 > 5.99 && result3 < 6.01);

    // Test 4: Float average - will fail due to restrictive Numeric concept
    std::cout << "Test 4: average(5.0f, 15.0f) = ";
    auto result4 = average(5.0f, 15.0f);
    std::cout << result4 << " (expected: 10.0)\n";
    assert(result4 > 9.99f && result4 < 10.01f);

    // Test 5: Vector of integers - container sum
    std::cout << "Test 5: sum(vector<int>{1,2,3,4,5}) = ";
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    auto result5 = container_sum(vec1);
    std::cout << result5 << " (expected: 15)\n";
    assert(result5 == 15);

    // Test 6: Vector of integers - average
    std::cout << "Test 6: average(vector<int>{1,2,3,4,5}) = ";
    auto result6 = container_average(vec1);
    std::cout << result6 << " (expected: 3)\n";
    assert(result6 == 3);

    // Test 7: Vector of integers - scale operation
    std::cout << "Test 7: scale(vector<int>{1,2,3,4,5}, 2) - modifies in place\n";
    scale(vec1, 2);
    std::cout << "Result: ";
    for (auto val : vec1) std::cout << val << " ";
    std::cout << "(expected: 2 4 6 8 10)\n";
    assert(vec1[0] == 2 && vec1[4] == 10);

    // Test 8: Vector of doubles - will fail due to Container concept issues
    std::cout << "Test 8: sum(vector<double>{1.5,2.5,3.5}) = ";
    std::vector<double> vec2 = {1.5, 2.5, 3.5};
    auto result8 = container_sum(vec2);
    std::cout << result8 << " (expected: 7.5)\n";
    assert(result8 > 7.49 && result8 < 7.51);

    // Test 9: Vector of doubles - average
    std::cout << "Test 9: average(vector<double>{1.5,2.5,3.5}) = ";
    auto result9 = container_average(vec2);
    std::cout << result9 << " (expected: 2.5)\n";
    assert(result9 > 2.49 && result9 < 2.51);

    // Test 10: Vector of doubles - scale
    std::cout << "Test 10: scale(vector<double>{1.5,2.5,3.5}, 3.0)\n";
    scale(vec2, 3.0);
    std::cout << "Result: ";
    for (auto val : vec2) std::cout << val << " ";
    std::cout << "(expected: 4.5 7.5 10.5)\n";
    assert(vec2[0] > 4.49 && vec2[0] < 4.51);

    // Test 11: Array of integers - sum
    std::cout << "Test 11: sum(array<int, 5>{10,20,30,40,50}) = ";
    std::array<int, 5> arr1 = {10, 20, 30, 40, 50};
    auto result11 = container_sum(arr1);
    std::cout << result11 << " (expected: 150)\n";
    assert(result11 == 150);

    // Test 12: Array of integers - average
    std::cout << "Test 12: average(array<int, 5>{10,20,30,40,50}) = ";
    auto result12 = container_average(arr1);
    std::cout << result12 << " (expected: 30)\n";
    assert(result12 == 30);

    // Test 13: Array of integers - scale
    std::cout << "Test 13: scale(array<int, 5>{10,20,30,40,50}, 2)\n";
    scale(arr1, 2);
    std::cout << "Result: ";
    for (auto val : arr1) std::cout << val << " ";
    std::cout << "(expected: 20 40 60 80 100)\n";
    assert(arr1[0] == 20 && arr1[4] == 100);

    // Test 14: List of floats - sum (will fail due to Container concept issues)
    std::cout << "Test 14: sum(list<float>{1.1,2.2,3.3,4.4}) = ";
    std::list<float> list1 = {1.1f, 2.2f, 3.3f, 4.4f};
    auto result14 = container_sum(list1);
    std::cout << result14 << " (expected: 11.0)\n";
    assert(result14 > 10.99f && result14 < 11.01f);

    // Test 15: List of floats - average
    std::cout << "Test 15: average(list<float>{1.1,2.2,3.3,4.4}) = ";
    auto result15 = container_average(list1);
    std::cout << result15 << " (expected: 2.75)\n";
    assert(result15 > 2.74f && result15 < 2.76f);

    // Test 16: List of floats - scale
    std::cout << "Test 16: scale(list<float>{1.1,2.2,3.3,4.4}, 2.0f)\n";
    scale(list1, 2.0f);
    std::cout << "Result: ";
    for (auto val : list1) std::cout << val << " ";
    std::cout << "(expected: 2.2 4.4 6.6 8.8)\n";
    
    // Test 17: Mixed signedness - should work with proper concepts
    std::cout << "Test 17: sum(10, -5) = ";
    auto result17 = sum(10, -5);
    std::cout << result17 << " (expected: 5)\n";
    assert(result17 == 5);

    // Test 18: Product of floating point values
    std::cout << "Test 18: multiply(2.5, 4.0) = ";
    auto result18 = multiply(2.5, 4.0);
    std::cout << result18 << " (expected: 10.0)\n";
    assert(result18 > 9.99 && result18 < 10.01);

    // Test 19: Vector with single element
    std::cout << "Test 19: sum(vector<int>{42}) = ";
    std::vector<int> vec3 = {42};
    auto result19 = container_sum(vec3);
    std::cout << result19 << " (expected: 42)\n";
    assert(result19 == 42);

    // Test 20: Empty-like operation (average of two same values)
    std::cout << "Test 20: average(100, 100) = ";
    auto result20 = average(100, 100);
    std::cout << result20 << " (expected: 100)\n";
    assert(result20 == 100);

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}