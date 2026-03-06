#!/bin/bash

cd /tmp/legacy_calc

rm -f configure Makefile.in Makefile config.* aclocal.m4
rm -rf autom4te.cache

cat > calculator.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int num1, num2, result;
    
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number1> <number2>\n", argv[0]);
        return 1;
    }
    
    num1 = atoi(argv[1]);
    num2 = atoi(argv[2]);
    result = num1 + num2;
    
    printf("%d\n", result);
    
    return 0;
}
EOF

cat > configure.ac << 'EOF'
AC_INIT([calculator], [1.0])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
EOF

cat > Makefile.am << 'EOF'
bin_PROGRAMS = calculator
calculator_SOURCES = calculator.c
EOF

autoreconf --install --force

./configure

make