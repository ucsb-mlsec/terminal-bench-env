#include "container.h"
#include <iostream>

void processFile1();
void processFile2();
void processFile3();

int main() {
    std::cout << "File1:" << std::endl;
    processFile1();
    std::cout << "File2:" << std::endl;
    processFile2();
    std::cout << "File3:" << std::endl;
    processFile3();
    return 0;
}