//go:build cgo

package processor

/*
#include <stdlib.h>
#include <string.h>

// C function to calculate string length and perform simple operation
int process_native(const char* input) {
    if (input == NULL) {
        return 0;
    }
    int len = strlen(input);
    return len * 2;
}

// C function to manipulate string
char* transform_string(const char* input) {
    if (input == NULL) {
        return NULL;
    }
    int len = strlen(input);
    char* result = (char*)malloc(len + 1);
    strcpy(result, input);
    return result;
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Process implements string processing using native C code
// This version uses CGO for performance-critical operations
func Process(input string) string {
	// Convert Go string to C string
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	// Call native C function
	length := C.process_native(cInput)
	cResult := C.transform_string(cInput)
	defer C.free(unsafe.Pointer(cResult))

	// Convert C string back to Go string
	goResult := C.GoString(cResult)

	// Return result indicating cgo-enabled processing
	return fmt.Sprintf("cgo-enabled: processed '%s' with native code (length metric: %d)", goResult, int(length))
}