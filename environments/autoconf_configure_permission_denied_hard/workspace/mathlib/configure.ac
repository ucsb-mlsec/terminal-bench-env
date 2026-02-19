AC_PREREQ([2.69])
AC_INIT([mathlib], [1.0], [support@mathlib.org])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_SRCDIR([src/mathops.c])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])

# Checks for programs
AC_PROG_CC
AC_PROG_INSTALL
AC_PROG_MAKE_SET
LT_INIT

# Checks for libraries
AC_CHECK_LIB([m], [sqrt])

# Checks for header files
AC_CHECK_HEADERS([math.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics
AC_TYPE_SIZE_T
AC_C_INLINE

# Checks for library functions
AC_CHECK_FUNCS([sqrt pow floor ceil])

AC_CONFIG_FILES([
    Makefile
    src/Makefile
])

AC_OUTPUT