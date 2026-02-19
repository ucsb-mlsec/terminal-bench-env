#include <iostream>
#include "container.h"

void processFile1() {
    Container<int> container;
    container.add(10);
    container.add(20);
    container.add(30);
    container.print();
}