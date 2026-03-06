#include <iostream>
#include "lib1/lib1.h"
#include "lib2/lib2.h"

int main() {
    lib1_function();
    lib2_function();
    std::cout << "Application completed successfully" << std::endl;
    return 0;
}