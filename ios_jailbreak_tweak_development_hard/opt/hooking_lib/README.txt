Function Hooking Library - Proof of Concept
=====================================================

PURPOSE:
--------
This library is a proof-of-concept implementation demonstrating iOS-style
method hooking techniques adapted for Linux systems. It uses LD_PRELOAD
to intercept standard C library function calls, allowing for runtime
function replacement and call interception. This PoC was developed to
test hooking mechanisms before deployment to iOS devices.

USAGE:
------
Compilation:
  cd /opt/hooking_lib
  make

Running with the library:
  LD_PRELOAD=/opt/hooking_lib/libhook.so ./test_program

The library will automatically intercept configured functions when loaded.

HOOKED FUNCTIONS:
-----------------
The following standard C library functions are currently intercepted:
  - strcpy: String copy operations
  - malloc: Memory allocation
  - free: Memory deallocation
  - printf: Formatted output

LIBRARY FILES:
--------------
  hooks.c          - Implementation of all hook functions
  hook_loader.c    - Library initialization and setup code
  libhook.so       - Compiled shared library (output)
  Makefile         - Build configuration

WARNING:
--------
This is experimental test code and should NOT be used in production
environments. The hook implementations may contain security vulnerabilities
and are intended for research and testing purposes only.