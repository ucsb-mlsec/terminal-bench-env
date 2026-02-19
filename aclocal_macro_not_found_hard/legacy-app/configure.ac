AC_INIT([legacy-app], [1.0], [maintainer@example.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_HEADERS([config.h])

# Check for required programs
AC_PROG_CC
AC_PROG_INSTALL

# Initialize libtool for shared library support
LT_INIT

# Check for required libraries using pkg-config
PKG_CHECK_MODULES([SQLITE3], [sqlite3 >= 3.0])

# Define compiler and linker flags
AC_SUBST([SQLITE3_CFLAGS])
AC_SUBST([SQLITE3_LIBS])

# Configure output files
AC_CONFIG_FILES([Makefile])

# Generate the output
AC_OUTPUT