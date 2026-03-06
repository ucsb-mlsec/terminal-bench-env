package main

/*
#cgo CFLAGS: -I/nonexistent/path
#include <stdlib.h>

int add(int a, int b) {
    return a + b;
}
*/
import "C"
import "fmt"

func main() {
    // Calculate the sum of 5 and 7 using C function
    num1 := C.int(5)
    num2 := C.int(7)
    
    result := C.add(num1, num2)
    
    fmt.Printf("The sum of 5 and 7 is: %d\n", int(result))
}