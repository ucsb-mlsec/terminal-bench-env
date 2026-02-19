AC_INIT([mathlib], [1.0], [dev@example.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AC_PROG_RANLIB

AC_CHECK_HEADERS([stdio.h stdlib.h math.h string.h])
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_FUNCS([malloc free sqrt pow sin cos])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT