# Custom M4 macros for legacy-project

dnl MY_CUSTOM_CHECK_LIB
dnl Check for a required library
AC_DEFUN([MY_CUSTOM_CHECK_LIB],
[
  AC_MSG_CHECKING([for $1 library])
  
  dnl Missing brackets around AC_CHECK_LIB arguments - ERROR 1
  AC_CHECK_LIB($1, $2,
    [
      AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_LIB]AS_TR_CPP($1), [1], 
                [Define to 1 if you have the $1 library.])
      LIBS="-l$1 $LIBS"
    ],
    [
      AC_MSG_RESULT([no])
      dnl Unbalanced brackets - ERROR 2
      AC_MSG_ERROR([Required library $1 not found. Please install lib$1-dev package.]
    )
  )
])

dnl MY_FEATURE_TEST
dnl Test compiler feature support
AC_DEFUN([MY_FEATURE_TEST],
[
  AC_MSG_CHECKING([whether compiler supports $1])
  
  dnl Save original flags
  SAVED_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS $1"
  
  dnl Missing brackets around program - ERROR 3
  AC_COMPILE_IFELSE(
    AC_LANG_PROGRAM([[
      #include <stdio.h>
    ]], [[
      printf("test");
      return 0;
    ]]),
    [
      AC_MSG_RESULT([yes])
      FEATURE_CFLAGS="$FEATURE_CFLAGS $1"
    ],
    [
      AC_MSG_RESULT([no])
      CFLAGS="$SAVED_CFLAGS"
    ]
  )
])