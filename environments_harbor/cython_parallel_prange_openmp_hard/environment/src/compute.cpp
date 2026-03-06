#include <Python.h>
#include <omp.h>
#include <vector>
#include <cmath>

// Compute parallel function that performs intensive calculations
static PyObject* compute_parallel(PyObject* self, PyObject* args) {
    PyObject* input_list;
    
    if (!PyArg_ParseTuple(args, "O", &input_list)) {
        return NULL;
    }
    
    if (!PyList_Check(input_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list");
        return NULL;
    }
    
    Py_ssize_t size = PyList_Size(input_list);
    std::vector<double> data(size);
    
    // Extract data from Python list
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(input_list, i);
        data[i] = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    
    std::vector<double> results(size);
    
    // Parallel computation with OpenMP
    #pragma omp parallel for
    for (Py_ssize_t i = 0; i < size; i++) {
        double value = data[i];
        double result = 0.0;
        
        // Intensive computation loop
        for (int j = 0; j < 1000; j++) {
            result += std::sin(value + j * 0.01) * std::cos(value - j * 0.01);
            result += std::sqrt(std::abs(value * j + 1.0));
            result = std::fmod(result, 1000.0);
        }
        
        results[i] = result;
    }
    
    // Sum all results
    double total = 0.0;
    for (size_t i = 0; i < results.size(); i++) {
        total += results[i];
    }
    
    return PyFloat_FromDouble(total);
}

// Check if parallel processing is enabled
static PyObject* is_parallel_enabled(PyObject* self, PyObject* args) {
    #ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (max_threads > 1) {
        Py_RETURN_TRUE;
    }
    #endif
    Py_RETURN_FALSE;
}

// Method definitions
static PyMethodDef ComputeMethods[] = {
    {"compute_parallel", compute_parallel, METH_VARARGS, 
     "Perform parallel computation on a list of numbers"},
    {"is_parallel_enabled", is_parallel_enabled, METH_NOARGS,
     "Check if OpenMP parallel processing is enabled"},
    {NULL, NULL, 0, NULL}
};

// Module definition for Python 3
static struct PyModuleDef computemodule = {
    PyModuleDef_HEAD_INIT,
    "compute",
    "Numerical computation module with OpenMP support",
    -1,
    ComputeMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_compute(void) {
    return PyModule_Create(&computemodule);
}