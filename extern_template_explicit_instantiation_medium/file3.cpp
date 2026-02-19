#include "container.h"
#include <iostream>

void processFile3() {
    Container<int> container;
    container.add(70);
    container.add(80);
    container.add(90);
    container.print();
}