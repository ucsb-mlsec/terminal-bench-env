#include <Python.h>
#include <string>
#include <algorithm>

// C++ function that correctly reverses a string
std::string reverse_string_impl(const std::string& input) {
    std::string result = input;
    std::reverse(result.begin(), result.end());
    return result;
}

// Python wrapper function for reverse_string
static PyObject* reverse_string(PyObject* self, PyObject* args) {
    const char* input;
    
    // Parse arguments - expecting a string
    if (!PyArg_ParseTuple(args, "s", &input)) {
        return NULL;
    }
    
    // Call the C++ implementation
    std::string result = reverse_string_impl(std::string(input));
    
    // Return the result as a Python string
    return PyUnicode_FromString(result.c_str());
}

// Method definition array - INTENTIONAL BUGS HERE
static PyMethodDef StringToolsMethods[] = {
    // BUG 1: Wrong method flags - using METH_NOARGS instead of METH_VARARGS
    {"reverse_string", reverse_string, METH_NOARGS, "Reverse a string"},
    
    // BUG 2: Missing NULL sentinel - this array should end with {NULL, NULL, 0, NULL}
    // Without it, Python will read beyond the array bounds causing crashes
};

// Module definition structure - INTENTIONAL BUGS HERE
static struct PyModuleDef stringtoolsmodule = {
    PyModuleDef_HEAD_INIT,
    // BUG 3: Wrong module name - doesn't match the actual module name expected
    "wrongname",  // Should be "stringtools"
    "String manipulation tools",
    -1,
    StringToolsMethods
};

// Module initialization function - INTENTIONAL BUGS HERE
// BUG 4: Wrong function name - doesn't follow PyInit_<modulename> convention
PyMODINIT_FUNC PyInit_wrongmodulename(void) {
    PyObject* module;
    
    module = PyModule_Create(&stringtoolsmodule);
    
    // BUG 5: Missing error checking
    // Should check if module is NULL before returning
    
    return module;
}