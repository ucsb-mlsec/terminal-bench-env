%module signal_module

%{
#include "signal_ops.h"
%}

%include "carrays.i"

/* Typemap for input arrays */
%typemap(in) (double *input, int size) {
    int i;
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list");
        return NULL;
    }
    $2 = PyList_Size($input);
    $1 = (double *) malloc($2 * sizeof(double));
    for (i = 0; i < $2; i++) {
        PyObject *o = PyList_GetItem($input, i);
        if (PyFloat_Check(o)) {
            $1[i] = PyFloat_AsDouble(o);
        } else if (PyInt_Check(o)) {
            $1[i] = (double)PyInt_AsLong(o);
        } else {
            free($1);
            PyErr_SetString(PyExc_TypeError, "List items must be numbers");
            return NULL;
        }
    }
}

/* Typemap for output arrays - missing freearg */
%typemap(out) double * {
    int i;
    $result = PyList_New(arg2);
    for (i = 0; i < arg2; i++) {
        PyList_SetItem($result, i, PyFloat_FromDouble($1[i]));
    }
    free($1);
}

/* Missing argout typemap for proper memory management */
%typemap(in, numinputs=0) double **output (double *temp) {
    $1 = &temp;
}

/* Incorrect syntax - missing closing brace */
%typemap(argout) double **output {
    int i;
    PyObject *o = PyList_New(arg2);
    for (i = 0; i < arg2; i++) {
        PyList_SetItem(o, i, PyFloat_FromDouble((*$1)[i]));
    
    $result = o;
}

/* Include the header - but missing the quotes/brackets */
%include signal_ops.h