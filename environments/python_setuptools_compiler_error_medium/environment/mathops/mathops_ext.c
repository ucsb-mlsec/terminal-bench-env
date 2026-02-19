#include <Python.h>

static PyObject* add(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    return PyLong_FromLong(a + b);
}

static PyMethodDef mathops_methods[] = {
    {"add", add, METH_VARARGS, "Add two integers"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mathops_module = {
    PyModuleDef_HEAD_INIT,
    "mathops",
    "Mathematical operations module",
    -1,
    mathops_methods
};

PyMODINIT_FUNC PyInit_mathops(void) {
    return PyModule_Create(&mathops_module);
}