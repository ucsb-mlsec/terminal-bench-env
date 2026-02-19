DATA PROCESSOR APPLICATION
=========================

PURPOSE:
This processor is a legacy data processing application designed to read and 
process structured input files. It performs line-by-line analysis and 
transformation of data according to predefined business rules.

USAGE:
Run the processor from the command line:
    ./processor <input_file>

Example:
    ./processor data.txt

DEPENDENCIES:
The processor requires the custom shared library libcustom.so.1 to be available
at runtime. This library contains essential functions for data transformation
and validation routines.

EXPECTED BEHAVIOR:
When executed successfully, the processor will:
- Read the input file line by line
- Apply transformation rules to each line
- Output processed results to standard output
- Exit with status 0 on successful completion

LIBRARY LOADING:
The shared library (libcustom.so.1) must be located where the dynamic linker
can find it. This can be achieved by:
- Placing the library in a system library path (/lib, /usr/lib, etc.)
- Setting the LD_LIBRARY_PATH environment variable to include the library directory
- Configuring /etc/ld.so.conf and running ldconfig (requires root access)

PREVIOUS DEPLOYMENT NOTES:
On the previous server infrastructure, libcustom.so.1 was installed in 
/usr/local/lib, which was configured as a standard library search path.
The LD_LIBRARY_PATH was also pre-configured in the system environment to
include custom library locations.

TROUBLESHOOTING:
If you encounter "error while loading shared libraries" errors, verify that
LD_LIBRARY_PATH includes the directory containing libcustom.so.1.