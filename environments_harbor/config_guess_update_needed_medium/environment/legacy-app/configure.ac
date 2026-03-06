AC_PREREQ([2.69])
AC_INIT([legacy-app], [1.0], [support@example.com])
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_AUX_DIR([build-aux])

# Detect the build system type
AC_CANONICAL_BUILD
AC_CANONICAL_HOST

# Checks for programs
AC_PROG_CC
AC_PROG_INSTALL

# Checks for libraries
AC_CHECK_LIB([m], [sqrt])

# Checks for header files
AC_CHECK_HEADERS([stdlib.h string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics
AC_TYPE_SIZE_T
AC_C_CONST

# Checks for library functions
AC_FUNC_MALLOC
AC_CHECK_FUNCS([memset strdup])

# Output files
AC_CONFIG_FILES([Makefile])
AC_OUTPUT