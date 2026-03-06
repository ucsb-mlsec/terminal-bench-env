#include <stddef.h>
#include "array_utils.h"

int array_max(int* array, size_t length) {
    if (length == 0) return 0;
    
    int max = array[0];
    for (size_t i = 1; i < length; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

long array_sum(int* array, size_t length) {
    long sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum;
}

void array_reverse(int* array, size_t length) {
    if (length <= 1) return;
    
    size_t left = 0;
    size_t right = length - 1;
    
    while (left < right) {
        int temp = array[left];
        array[left] = array[right];
        array[right] = temp;
        left++;
        right--;
    }
}