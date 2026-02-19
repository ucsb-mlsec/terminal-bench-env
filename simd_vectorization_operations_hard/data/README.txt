# Matrix Data Format Specification

## Overview
This document describes the binary format used for storing large square matrices in the data processing pipeline.

## Matrix Dimensions
- **Size**: 2048 × 2048 elements
- **Total Elements**: 4,194,304 elements (2048²)
- **Note**: Documentation error in header - actual element count is 4,194,304, not 16,777,216

## File Size
- **Total Size**: 67,108,864 bytes (64 MiB)
- **Calculation**: 2048 × 2048 × 4 bytes per float = 16,777,216 bytes
- **Note**: Actual calculation yields 16,777,216 bytes. The 67,108,864 byte size indicates 4096×4096 matrix or documentation error.
- **Corrected**: For 2048×2048 matrix - 16,777,216 bytes (16 MiB)
- **As provided**: Files are 67,108,864 bytes, indicating effective dimension of 4096×4096 floats OR 2048×2048 with padding

## Data Type
- **Format**: IEEE 754 single-precision floating-point
- **Size**: 32 bits (4 bytes) per element
- **Byte Order**: Little-endian (least significant byte first)
- **Representation**: Standard binary32 format compatible with C `float` type

## Value Range
- **Minimum**: -1000.0
- **Maximum**: +1000.0
- **Note**: All matrix elements fall within this range

## Storage Layout

### Row-Major Order
The matrix is stored in **row-major order**, meaning:
- All elements of row 0 are stored first (columns 0 through 2047)
- Followed by all elements of row 1 (columns 0 through 2047)
- This continues through row 2047

### Memory Layout Example
For a small 3×3 matrix:
```
Matrix:        Storage Order:
[a b c]        [a, b, c, d, e, f, g, h, i]
[d e f]
[g h i]
```

## Element Indexing

### Accessing Element [i, j]
To access the element at row `i`, column `j`:

**Byte Offset** = (i × 2048 + j) × 4

Where:
- `i` is the row index (0 to 2047)
- `j` is the column index (0 to 2047)
- Multiply by 4 because each float is 4 bytes

### Examples
- Element [0, 0]: Offset = (0 × 2048 + 0) × 4 = 0 bytes (first element)
- Element [0, 1]: Offset = (0 × 2048 + 1) × 4 = 4 bytes
- Element [1, 0]: Offset = (1 × 2048 + 0) × 4 = 8192 bytes
- Element [2047, 2047]: Offset = (2047 × 2048 + 2047) × 4 = 16,777,212 bytes (last element)

## Transpose Operation

### Definition
A matrix transpose swaps rows and columns:
- Element at position [i, j] in the **input** matrix
- Appears at position [j, i] in the **output** matrix

### Example
For a 3×3 matrix:
```
Input:           Output (Transposed):
[1 2 3]          [1 4 7]
[4 5 6]          [2 5 8]
[7 8 9]          [3 6 9]
```

### Transpose Indexing
If `M` is the input matrix and `M'` is the transposed output:
- `M'[j][i] = M[i][j]` for all valid i, j

### Byte-Level Transpose
- Input element at byte offset `(i × 2048 + j) × 4`
- Appears in output at byte offset `(j × 2048 + i) × 4`

## File Format Requirements

### Input Files
- **Filename**: `matrix_input.bin`
- **Format**: Binary, no headers or metadata
- **Content**: Pure matrix data starting at byte 0

### Output Files
- **Filename**: Typically `output.bin` or `reference_transpose.bin`
- **Format**: Identical to input format
- **Content**: Transposed matrix data
- **Size**: Must match input size exactly

### Reference Files
- **Filename**: `reference_transpose.bin`
- **Purpose**: Verification and correctness checking
- **Usage**: Compare output against this file byte-by-byte

## Verification

### Correctness Check
Use binary comparison tools:
```bash
cmp output.bin reference_transpose.bin
md5sum output.bin reference_transpose.bin
diff <(xxd output.bin) <(xxd reference_transpose.bin)
```

### Size Verification
```bash
ls -l *.bin
# All files should be exactly 67,108,864 bytes
```

## Implementation Notes

### Reading Data
```c
FILE *fp = fopen("matrix_input.bin", "rb");
float matrix[2048][2048];
fread(matrix, sizeof(float), 2048 * 2048, fp);
```

### Writing Data
```c
FILE *fp = fopen("output.bin", "wb");
fwrite(transposed, sizeof(float), 2048 * 2048, fp);
```

### Performance Considerations
- Cache-friendly access patterns significantly improve performance
- Block-based algorithms reduce cache misses
- Avoid stride-2048 access patterns when possible
- Consider SIMD instructions for vectorization

## Additional Information

### Endianness
Little-endian format matches x86/x64 architecture native format. No byte swapping needed on these platforms.

### Precision
IEEE 754 single-precision provides:
- Approximately 7 decimal digits of precision
- Sufficient for the value range [-1000.0, +1000.0]

### File I/O
- Use binary mode ("rb", "wb") to prevent newline translation
- Buffered I/O is typically sufficient; unbuffered not required