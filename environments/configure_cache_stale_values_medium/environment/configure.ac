AC_INIT([myapp], [1.0])
AC_PROG_CC

# Check for SQLite3 library
AC_CHECK_LIB([sqlite3], [sqlite3_open], [],
    [AC_MSG_ERROR([SQLite3 library not found])])

# Check for SQLite3 header
AC_CHECK_HEADERS([sqlite3.h], [],
    [AC_MSG_ERROR([SQLite3 header not found])])

# Specify output files
AC_CONFIG_FILES([Makefile])

AC_OUTPUT