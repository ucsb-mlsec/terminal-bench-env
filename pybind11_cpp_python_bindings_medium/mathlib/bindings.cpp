#include <pybind11/pybind11.h>
#include "math_functions.h"

namespace py = pybind11;

PYBIND11_MODULE(mathlib_cpp, m) {
    m.doc() = "Python bindings for mathematical library";

    // Binding for add function - MISSING SEMICOLON
    m.def("add", &add, "Add two numbers")
    
    // Binding for subtract function - WRONG FUNCTION NAME (typo)
    m.def("subtract", &subtrct, "Subtract two numbers");
    
    // Binding for multiply function - INCORRECT DEF SYNTAX (missing description argument)
    m.def("multiply", &multiply);
    
    // Binding for divide function
    m.def("divide", &divide, "Divide two numbers");
    
    // Binding for power function
    m.def("power", &power, "Raise a number to a power");
}