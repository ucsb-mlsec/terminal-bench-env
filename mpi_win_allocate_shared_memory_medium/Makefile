CC=mpicc
CFLAGS=-O2 -Wall

all: data_processor data_processor_shared

data_processor: data_processor.c
	$(CC) $(CFLAGS) -o data_processor data_processor.c

data_processor_shared: data_processor_shared.c
	$(CC) $(CFLAGS) -o data_processor_shared data_processor_shared.c

clean:
	rm -f data_processor data_processor_shared *.o