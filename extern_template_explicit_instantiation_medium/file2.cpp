#include "container.h"
#include <iostream>

void processFile2() {
    Container<int> container;
    container.add(40);
    container.add(50);
    container.add(60);
    container.print();
}