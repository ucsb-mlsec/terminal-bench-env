#!/usr/bin/env python3

import json
from cffi import FFI

def read_matrix(filename):
    """Read matrix from file in the specified format."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # First line contains dimensions
    rows, cols = map(int, lines[0].strip().split())
    
    # Read matrix values
    matrix = []
    for i in range(1, rows + 1):
        row = list(map(float, lines[i].strip().split()))
        matrix.extend(row)
    
    return rows, cols, matrix

def write_matrix(filename, rows, cols, matrix):
    """Write matrix to file in the specified format."""
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        idx = 0
        for i in range(rows):
            row_values = []
            for j in range(cols):
                row_values.append(str(matrix[idx]))
                idx += 1
            f.write(' '.join(row_values) + '\n')

def main():
    # Create CFFI interface
    ffi = FFI()
    
    # BUGGY: Wrong type (int* instead of double*) and wrong function name
    ffi.cdef('void matrix_multiply(int* a, int a_rows, int a_cols, int* b, int b_rows, int b_cols, int* result);')
    
    # BUGGY: Wrong path - using relative path instead of absolute
    lib = ffi.dlopen('./libmatrix.so')
    
    # Read input matrices
    a_rows, a_cols, matrix_a = read_matrix('/workspace/matrix_lib/matrix_a.txt')
    b_rows, b_cols, matrix_b = read_matrix('/workspace/matrix_lib/matrix_b.txt')
    
    # Verify dimensions are compatible
    if a_cols != b_rows:
        raise ValueError(f"Matrix dimensions incompatible: {a_rows}x{a_cols} and {b_rows}x{b_cols}")
    
    # Prepare arrays for C
    a_array = ffi.new("double[]", matrix_a)
    b_array = ffi.new("double[]", matrix_b)
    
    # Result matrix will be a_rows x b_cols
    result_rows = a_rows
    result_cols = b_cols
    result_array = ffi.new("double[]", result_rows * result_cols)
    
    # Call C function
    lib.matrix_multiply(a_array, a_rows, a_cols, b_array, b_rows, b_cols, result_array)
    
    # Convert result back to Python list
    result_matrix = [result_array[i] for i in range(result_rows * result_cols)]
    
    # Write result to file
    write_matrix('/workspace/result.txt', result_rows, result_cols, result_matrix)
    
    # Create status file
    status = {
        "completed": True,
        "result_file": "/workspace/result.txt"
    }
    
    with open('/workspace/status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"Matrix multiplication completed successfully!")
    print(f"Result: {result_rows}x{result_cols} matrix saved to /workspace/result.txt")

if __name__ == '__main__':
    main()