#include <iostream>
#include <vector>
#include <list>
#include <string>
#include "include/container_traits.hpp"

using namespace container_utils;

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::list<std::string> lst = {"hello", "world"};
    
    std::cout << std::boolalpha;
    
    std::cout << "Vector size: " << container_size(vec) << std::endl;
    std::cout << "List size: " << container_size(lst) << std::endl;
    
    std::cout << "Vector is empty: " << is_empty(vec) << std::endl;
    std::cout << "List is empty: " << is_empty(lst) << std::endl;
    
    std::vector<double> empty_vec;
    std::cout << "Empty vector is empty: " << is_empty(empty_vec) << std::endl;
    
    return 0;
}