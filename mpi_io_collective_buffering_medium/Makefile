all: data_writer

data_writer: data_writer.c
	mpicc -O2 -o data_writer data_writer.c

clean:
	rm -f data_writer /tmp/output.dat