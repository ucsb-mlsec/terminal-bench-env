AC_INIT([legacy-project], [1.0], [bugs@example.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_HEADERS([config.h])

# Check for programs
AC_PROG_CC
AC_PROG_INSTALL

# Check for standard headers
AC_CHECK_HEADERS([stdlib.h string.h unistd.h])

# Check for library functions
AC_CHECK_FUNCS([malloc memset strdup])

# Custom macro calls with quoting errors
MY_CUSTOM_CHECK_LIB(pthread, pthread_create, [
    AC_DEFINE([HAVE_PTHREAD], [1], [Define if pthread is available])
], [
    AC_MSG_ERROR([pthread library not found])
])

# Unbalanced quotes in macro call
MY_FEATURE_TEST([networking], [enable-network, 
    AC_DEFINE([ENABLE_NETWORK], [1], [Enable networking support])
)

# Check for pkg-config
PKG_CHECK_MODULES([DEPS], [glib-2.0 >= 2.40])

# Output files
AC_CONFIG_FILES([Makefile])
AC_OUTPUT