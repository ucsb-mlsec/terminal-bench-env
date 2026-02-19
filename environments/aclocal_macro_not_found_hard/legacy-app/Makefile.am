bin_PROGRAMS = legacy-app

legacy_app_SOURCES = main.c mathops.c mathops.h dbops.c dbops.h

legacy_app_CFLAGS = $(SQLITE3_CFLAGS)
legacy_app_LDADD = $(SQLITE3_LIBS)

ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = -Wall -Wextra