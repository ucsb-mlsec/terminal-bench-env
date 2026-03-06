//go:build cgo

package main

/*
#include <stdlib.h>

int enhance(int x) {
    return x * 3 + 7;
}
*/
import "C"

func Calculate(n int) int {
	result := C.enhance(C.int(n))
	return int(result)
}