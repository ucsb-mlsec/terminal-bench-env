AC_PREREQ([2.69])
AC_INIT([broken-project], [1.0], [dev@example.com])
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])

AM_INIT_AUTOMAKE([-Wall -Werror foreign])

AC_PROG_CC
AC_PROG_INSTALL

dnl Error Type 1: Malformed AC_CHECK_LIB - missing function parameter
AC_CHECK_LIB([m])

dnl Check for pthread library properly
AC_CHECK_LIB([pthread], [pthread_create])

dnl Error Type 3: Unquoted macro expansion
CFLAGS=$CFLAGS -Wall

dnl Call custom macro
CUSTOM_CHECK_FEATURE

dnl Another unquoted expansion
PREFIX_DIR=$prefix

AC_CONFIG_FILES([
  Makefile
  src/Makefile
])

AC_OUTPUT