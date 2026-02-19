CC=clang
CFLAGS=-fprofile-instr-generate -fcoverage-mapping

all: test_calculator

test_calculator: test_calculator.c calculator.c
	$(CC) $(CFLAGS) -o test_calculator test_calculator.c calculator.c

clean:
	rm -f test_calculator *.profraw *.profdata